import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hparams as hp
import os

import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=hp.train_visible_devices

import numpy as np
import argparse
import time
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils
import audio as Audio

def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    
    # Get dataset
    dataset = Dataset("train.txt") 
    # !! 여기서 shuffle이 True여서 인덱스를 랜덤으로 부르는 듯
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, 
        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # initial 단계에서 speaker 추가
    # Define model

    # !! 파라미터로 n_speakers 추가, speaker_table은 spker_id와 동일한 역할
    # !! 둘다 tensor 형태로 바꿔주어야 함
    # 영어가 들어가면 텐서로 바꿀 수 없음....
    n_speakers, speaker_table = utils.get_speakers()
    n_speakers = torch.tensor(n_speakers).to(device)

    # !! 글자들을 바꾸기가 어려운지 텐서 변경이 안됨
    #speaker_table = torch.tensor(speaker_table)

    model = nn.DataParallel(FastSpeech2(n_speakers=n_speakers)).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # read params
    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)


    # Load vocoder
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path = hp.vocoder_pretrained_model_path)
        vocoder.to(device)
    elif hp.vocoder == 'hifigan':
        vocoder = utils.get_hifigan(ckpt_path = hp.vocoder_pretrained_model_path)
        vocoder.to(device)
    else:
        vocoder = None

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    # Training
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        # !! loader을 부를 때 getitems를 수행함
        # batch size는 설정한 batch의 제곱임
        for i, batchs in enumerate(loader):
            # btachs는 64개 중 8개를 뽑은 것
            for j, data_of_batch in enumerate(batchs):
                # !!! 메타데이터가 나쁘게 작성된 데이터에 대한 배치단위로 넘기기로 해결 -> 해결 더 필요
                if type(data_of_batch) == bool:
                    continue
                start_time = time.perf_counter()
                current_step = i*hp.batch_size + j + args.restore_step + epoch*len(loader)*hp.batch_size + 1
                
                # 이번 배치의 이번 차례의 데이터들
                # 여기에 spker_ids도 추가되어야 함 (07_M_LYW00_64 같은거)
                # Get Data
                #speaker_ids = torch.tensor(data_of_batch["id"]).long().to(device)
                ids = []
                for t in data_of_batch["id"]:
                    ids.append(int(t))
                speaker_ids = torch.tensor(ids).long().to(device)
                #print(speaker_ids)
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)

                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                # Forward
                # !! speaker_ids를 추가해 주어야 함
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, speaker_ids, mel_len, D, f0, energy, max_src_len, max_mel_len)
                
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                 
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")
                with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
                    f_f_loss.write(str(f_l)+"\n")
                with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
                    f_e_loss.write(str(e_l)+"\n")
                 
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()


                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "Time Used: {:.3f}s({:.1f}min), Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), ((Now-Start)/60), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    
                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                train_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                train_logger.add_scalar('Loss/energy_loss', e_l, current_step)
                
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))
                    
                    print(datetime.datetime.now() + datetime.timedelta(hours=9))

                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step, vocoder)
                        t_l = d_l + f_l + e_l + m_l + m_p_l
                        
                        print(datetime.datetime.now() + datetime.timedelta(hours=9))
                        val_logger.add_scalar('Loss/total_loss', t_l, current_step)
                        val_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                        val_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                        val_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                        val_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                        val_logger.add_scalar('Loss/energy_loss', e_l, current_step)

                    model.train()
                    
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
