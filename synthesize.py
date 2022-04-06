import torch
import torch.nn as nn
import numpy as np
import hparams as hp
import os

from utils import get_speakers


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.synth_visible_devices

import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2
from vocoder import hifigan_generator

from text import text_to_sequence, sequence_to_text
import utils
import audio as Audio

import codecs
#from g2pk import G2p
from jamo import h2j

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def kor_preprocess(text):
    text = text.rstrip(punctuation)
    
    #g2p=G2p()
    phone = text # !!! G2P 적용 X 
    #phone = g2p(text) # !!! G2P 적용 O
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)

# !! get FastSpeech에서도 n_speaker를 추가해 주어야 함
# !!!!! n_speakers 필요 없는데 이미 ckpt에서 사용했어서 필요함..
def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, text, sentence, prefix=''):
    sentence = sentence[:20] # long filename will result in OS Error

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    # Multi-speaker 테스트면 그냥 제일 첫 번째 사람을 지목,
    # single-speaker의 경우에는 제일 앞 본인을 선택할 수 있도록 ids[0]
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, synthesize=True)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    # Griffin lim은 잠시 들어가라
    #Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
    utils.hifigan_infer(mel_postnet_torch, path=os.path.join(hp.test_path, '{}_{}.wav'.format(sentence, prefix)))   
    if not os.path.exists(hp.test_path + '/plot'):
        os.mkdir(hp.test_path + '/plot')
    utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, 'plot/{}_{}.png'.format(sentence, prefix)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default=80000)
    args = parser.parse_args()

    # !!!!! 이부분 코드 너무 잘못 짬 추후 수정 필요
    n_speakers, _ = utils.get_speakers()
    n_speakers = torch.tensor(n_speakers).to(device)

    model = get_FastSpeech2(args.step).to(device)

    #kss
    eval_sentence=['그는 괜찮은 척하려고 애쓰는 것 같았다','그녀의 사랑을 얻기 위해 애썼지만 헛수고였다','용돈을 아껴써라','그는 아내를 많이 아낀다','요즘 공부가 안돼요','한 여자가 내 옆에 앉았다']
    train_sentence=['가까운 시일 내에 한번, 댁으로 찾아가겠습니다','우리의 승리는 기적에 가까웠다','아이들의 얼굴에는 행복한 미소가 가득했다','헬륨은 공기보다 가볍다','이것은 간단한 문제가 아니다']
    test_sentence='안녕하세요 반갑습니다 테스트랍니다'
    
    #g2p=G2p()
    print('which sentence do you want?')
    print('1.eval_sentence 2.train_sentence 3.test_sentence 4.create new sentence')

    mode=input()
    print('you went for mode {}'.format(mode))
    if mode=='4':
        print('input sentence')
        sentence = input()
    elif mode=='1':
        sentence = eval_sentence
    elif mode=='2':
        sentence = train_sentence
    elif mode=='3':
        sentence = test_sentence
    else:
        exit()
    
    print('sentence that will be synthesized: ')
    print(sentence)
    if mode == '1' or mode== '2':
        for sent in sentence:
            text = kor_preprocess(sent)
            synthesize(model, text, sent, prefix='step_{}'.format(args.step))
    else:
        text = kor_preprocess(sentence)
        synthesize(model, text, sentence, prefix='step_{}'.format(args.step))
