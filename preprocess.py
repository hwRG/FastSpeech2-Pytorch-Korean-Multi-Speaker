import os
from data import data_processing
import hparams as hp
import scipy.io as sio

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path
    meta = hp.meta_name
    textgrid_name = hp.textgrid_name

    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)

    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)

    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)

    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)

    #if os.path.isfile(textgrid_name):
    #    os.system('cp ./{} {}'.format(textgrid_name, out_dir))
    
    if not os.path.exists(os.path.join(out_dir, textgrid_name.replace(".zip", ""))):
        os.system('unzip {} -d {}'.format(textgrid_name, out_dir))
        
    speakers = os.listdir(os.path.join(in_dir, 'wavs'))
    sample_data = os.listdir(os.path.join(in_dir, 'wavs', speakers[0]))
    sample_rate, _ = sio.wavfile.read(os.path.join(in_dir, 'wavs', speakers[0], sample_data[1]))
    
    
    # Sampling rate가 다를 경우 원래 파일을 sampling rate 폴더명으로 변경하고 새로 16000인 wav 폴더 생성 
    if sample_rate != 16000:
        os.system('mv ' + in_dir + '/wavs ' + in_dir + '/wavs_{}'.format(str(sample_rate)))

        for speaker in speakers:
            dir_list = os.listdir(os.path.join(in_dir, 'wavs_{}'.format(str(sample_rate)), speaker))
            wav_list = []
            for dir in dir_list:
                if '.wav' in dir:
                    wav_list.append(dir)

            for wav in wav_list:
                wav_path = os.path.join(in_dir, 'wavs', speaker, wav)
                wav_before_path = os.path.join(in_dir, 'wavs_{}'.format(str(sample_rate)), speaker, wav)

                if not os.path.exists(os.path.join(in_dir, 'wavs')):
                    os.mkdir(os.path.join(in_dir, 'wavs'))
                if not os.path.exists(os.path.join(in_dir, 'wavs', speaker)):
                    os.mkdir(os.path.join(in_dir, 'wavs', speaker))

                os.system("ffmpeg -i {} -ac 1 -ar 16000 {}".format(wav_before_path, wav_path))

    train, val = data_processing.build_from_path(in_dir, out_dir, meta)

    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    main()
