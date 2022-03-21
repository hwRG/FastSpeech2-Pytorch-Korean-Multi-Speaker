import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from scipy.io import wavfile
from vocoder.hifigan_generator import Generator
import hparams as hp
import os
import text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# !! Speaker 불러오기
def get_speakers():
    path = 'preprocessed/' + hp.dataset + '/alignment'
    file_list = os.listdir(path)
    n_speakers = len(file_list)

    print('\nSpeaker Count:', n_speakers)

    return n_speakers, file_list


# !! Embedding 레이어 추가
def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    #m.weight.data.normal_(0, std)
    return m

# !! 스피커를 하나로 통합하는 모듈 구현 
class SpeakerIntegrator(nn.Module):

    def __init__(self):
        super(SpeakerIntegrator, self).__init__()

    def forward(self, x, spembs):
        """
        x      shape : (batch, 39, 256)
        spembs shape : (batch, 256)
        """
        spembs = spembs.unsqueeze(1)
        spembs = spembs.repeat(1, x.shape[1], 1)
        x = x + spembs
    
        return x


def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, np.array(durations), start_time, end_time

def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        spectrogram, pitch, energy = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
        
        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color='tomato')
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(0, hp.f0_max)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
        
        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(energy, color='darkviolet')
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
    
    plt.savefig(filename, dpi=200)
    plt.close()

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask

def get_vocgan(ckpt_path, n_mel_channels=hp.n_mel_channels, generator_ratio = [4, 4, 2, 2, 2, 2], n_residual_layers=4, mult=256, out_channels=1): 

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = Generator(n_mel_channels, n_residual_layers,
                        ratios=generator_ratio, mult=mult,
                        out_band=out_channels)

    model.load_state_dict(checkpoint['model_g'])
    model.to(device).eval()

    return model


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_hifigan(ckpt_path): 
    #checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict_g = load_checkpoint(ckpt_path, device)
    model = Generator().to(device)

    model.load_state_dict(state_dict_g['generator'], strict=False)
    
    #model.eval()

    return model


def vocgan_infer(mel, vocoder, path):
    model = vocoder

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)

        audio = model.infer(mel).squeeze()
        audio = hp.max_wav_value * audio[:-(hp.hop_length*10)]
        audio = audio.clamp(min=-hp.max_wav_value, max=hp.max_wav_value-1)
        audio = audio.short().cpu().detach().numpy()

        wavfile.write(path, hp.sampling_rate, audio)

# 아예 hifigan inference를 가져옴
def hifigan_infer(mel, path):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator().to(device)
    state_dict_g = load_checkpoint(hp.vocoder_pretrained_model_path, device)
    generator.load_state_dict(state_dict_g['generator'], strict=False)

    #filelist = os.listdir(.input_wavs_dir)

    #os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        #wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
        #wav = wav / MAX_WAV_VALUE
        #wav = torch.FloatTensor(mel).to(device)
        #x = get_mel(wav.unsqueeze(0))
        x = mel
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0 #(MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        wavfile.write(path, hp.sampling_rate, audio)
        print(path, 'done')


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

# from dathudeptrai's FastSpeech2 implementation
def standard_norm(x, mean, std, is_mel=False):

    if not is_mel:
        x = remove_outlier(x)

    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x

    return (x - mean) / std

def de_norm(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = mean + std * x
    x[zero_idxs] = 0.0
    return x


def _is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    return np.logical_or(x <= lower, x >= upper)

# old_man f0를 싹다 outlier로 판단해버림
def remove_outlier(x):
    """Remove outlier from x."""
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    
    indices_of_outliers = []

    for ind, value in enumerate(x):
        if _is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0

    # replace by mean f0.
    x[indices_of_outliers] = np.max(x)
    return x

def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


## HiFi-GAN use fuction

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)