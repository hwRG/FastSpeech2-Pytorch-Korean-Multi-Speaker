import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment, standard_norm, remove_outlier, average_by_duration
import hparams as hp
from jamo import h2j
import codecs
import random

from sklearn.preprocessing import StandardScaler

def prepare_align(in_dir, meta):
    with open(os.path.join(in_dir, meta), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename, text = parts[0], parts[3]

            basename=basename.replace('.wav','.txt')
            
            with open(os.path.join(in_dir,'wavs',basename),'w') as f1:
                f1.write(text)

def build_from_path(in_dir, out_dir, meta):
    train, val = list(), list()

    scalers = [StandardScaler(copy=False) for _ in range(3)]	# scalers for mel, f0, energy
    n_frames = 0
    
    with open(os.path.join(in_dir, meta), encoding='utf-8') as f:
        for index, line in enumerate(f):
            parts = line.strip().split('|')
            basename, text = parts[0], parts[3]

            # Alignment로 Preprocessing checkpoint
            preCheck = out_dir + '/alignment/' + '{}-ali-{}.npy'.format(hp.dataset, basename[:-4])
            #print(preCheck)

            if os.path.isfile(preCheck):
                continue
            ret = process_utterance(in_dir, out_dir, basename, scalers)
            
            
            if ret is None or ret is False:
                continue
            else:
                info, n = ret
            
            #if basename[0] == '1':
            #    val.append(info)
            #else:
            #    train.append(info)
            
            rand = random.randrange(0,4)
            if rand < 1:
                val.append(info)
            else:
                train.append(info)


            if index % 100 == 0:
                print("Done %d" % index)

            n_frames += n
            
    param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
    print(param_list)
    param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
    [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename, scalers):
    wav_bak_basename=basename.replace('.wav','')

    basename = wav_bak_basename # old_man_city처럼 한 폴더에 다 있을 경우
    # basename = wav_bak_basename[2:] KSS처럼 폴더가 있을 경우

    wav_bak_path = os.path.join(in_dir, "wavs_bak", "{}.wav".format(wav_bak_basename))
    wav_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))

    # Convert kss data into PCM encoded wavs
    if not os.path.isfile(wav_path):
        os.system("ffmpeg -i {} -ac 1 -ar 22050 {}".format(wav_bak_path, wav_path))    

    # Textgrid hparam에 의존하도록 변경
    #tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) 
    tg_path = os.path.join(out_dir, hp.textgrid_name.replace('.zip', ''), '{}.TextGrid'.format(basename)) 

    # TextGrid가 없을 경우
    if os.path.isfile(tg_path) == False:
        print(basename, 'TextGrid is Not Found.')
        return None

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]
    

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]

    # old_man에서 f0의 인덱스를 모두 선택해서 다 remov해버리는 문제가 발생함
    #f0, energy = remove_outlier(f0), remove_outlier(energy)

    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
    
    # f0[f0!=0]은 요소 안에 0을 뺀 모든 것을 의미
    if len(f0[f0!=0]) == 0 :
        print(basename, 'f0 error')
        return None

    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
   
    # 스케일러 (사용하는 건지 잘 모르겠음)
    mel_scaler, f0_scaler, energy_scaler = scalers
    mel_scaler.partial_fit(mel_spectrogram.T)
    f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
    energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    return '|'.join([basename, text]), mel_spectrogram.shape[1]
