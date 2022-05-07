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
import tqdm
import scipy.io as sio

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
            # !! basename은 100063/10006312229.wav 형식임
            # basename[0] : 폴더명 / basename[1] : 파일명 
            basename = parts[0].strip().split('/')
            text = parts[3]

            # Alignment로 Preprocessing checkpoint
            preCheck = out_dir + '/mel/' + basename[0] + '/{}-mel-{}.npy'.format(hp.dataset, basename[1][:-4])
            if os.path.isfile(preCheck):
                continue

            ret = process_utterance(in_dir, out_dir, basename, scalers)
            
            # !!! 시간이 너무 작아 f0가 없는 것과 Textgrid가 없는 등 상황에서 포함시키지 않음
            # 또한 8초 이상의 데이터는 학습에 방해가 될 수 있기 때문에 제거
            if float(parts[4]) > 9:
                print(parts[0], parts[4])
            if ret is None or ret is False or float(parts[4]) > 8:
                continue
            else:
                info, n = ret
            
            # train / val 나누는 과정 9:1로 진행
            rand = random.randrange(0,10)
            if rand < 1:
                val.append(info)
            else:
                train.append(info)


            if index % 1000 == 0:
                print("Done %d" % index)

            n_frames += n
            
    param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
    param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
    [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename, scalers):
    basename[1]=basename[1].replace('.wav','')

    wav_path = os.path.join(in_dir, 'wavs', basename[0], '{}.wav'.format(basename[1]))
    
    # preprocess.py로 이사했습니다
    """
    # !!! 만약 sampling rate가 16000이면 그대로 wavs로 변경되고 아니면
    # !!! 다른 sampling rate면 16000으로 변경되고 변경된 wav는 wavs 폴더로 저장, lab은 labs에 저장
    sample_rate, _ = sio.wavfile.read(wav_path)
    # Convert kss data into PCM encoded wavs
    if sample_rate != 16000:
        os.system('mv ' + in_dir + '/wavs ' + in_dir + '/wavs_{}'.format(str(sample_rate)))
        wav_before_path = os.path.join(in_dir, 'wavs_{}'.format(str(sample_rate)), basename[0], '{}.wav'.format(basename[1]))
        if not os.path.exists(os.path.join(in_dir, 'wavs')):
            os.mkdir(os.path.join(in_dir, 'wavs'))
        if not os.path.exists(os.path.join(in_dir, 'wavs', basename[0])):
            os.mkdir(os.path.join(in_dir, 'wavs', basename[0]))
        os.system("ffmpeg -i {} -ac 1 -ar 16000 {}".format(wav_before_path, wav_path))
    """

    # Textgrid hparam에 의존하도록 변경
    #tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) 
    tg_path = os.path.join(out_dir, hp.textgrid_name.replace('.zip', ''), basename[0], '{}.TextGrid'.format(basename[1])) 

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

    # !! 2초 미만의 오디오가 아예 사라져 버리는 문제가 발생
    # !!! 시간도 오래걸리고, 짧은 데이터도 소중하기 때문에 주석처리
    #f0, energy = remove_outlier(f0), remove_outlier(energy)

    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)
    
    # f0[f0!=0]은 요소 안에 0을 뺀 모든 것을 의미
    if len(f0[f0!=0]) == 0 :
        print(basename, 'f0 error')
        return None

    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # !! 사용할 루트를 폴더명에 따라 저장함 
    # Save alignment
    
    if not os.path.exists(os.path.join(out_dir, 'alignment', basename[0])):
        os.makedirs(os.path.join(out_dir, 'alignment', basename[0]), exist_ok=True)
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename[1])
    np.save(os.path.join(out_dir, 'alignment', basename[0], ali_filename), duration, allow_pickle=False)

    
    if not os.path.exists(os.path.join(out_dir, 'f0', basename[0])):
        os.makedirs(os.path.join(out_dir, 'f0', basename[0]), exist_ok=True)
    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename[1])
    np.save(os.path.join(out_dir, 'f0', basename[0], f0_filename), f0, allow_pickle=False)


    if not os.path.exists(os.path.join(out_dir, 'energy', basename[0])):
        os.makedirs(os.path.join(out_dir, 'energy', basename[0]), exist_ok=True)
    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename[1])
    np.save(os.path.join(out_dir, 'energy', basename[0], energy_filename), energy, allow_pickle=False)


    if not os.path.exists(os.path.join(out_dir, 'mel', basename[0])):
        os.makedirs(os.path.join(out_dir, 'mel', basename[0]), exist_ok=True)
    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename[1])
    np.save(os.path.join(out_dir, 'mel', basename[0], mel_filename), mel_spectrogram.T, allow_pickle=False)
   
    # 스케일러 (사용하는 건지 잘 모르겠음)
    mel_scaler, f0_scaler, energy_scaler = scalers
    mel_scaler.partial_fit(mel_spectrogram.T)
    f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
    energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    # !! 여기 애매한 듯
    return '|'.join([basename[1], text]), mel_spectrogram.shape[1]
