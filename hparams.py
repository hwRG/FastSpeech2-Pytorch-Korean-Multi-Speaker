import os

# 데이터셋 이름 정하는 것으로 dataset 안에 폴더 하나만 있어야 함
if os.path.exists(os.listdir('dataset')[0]):
    dataset = os.listdir('dataset')[0]
else:
    dataset = os.listdir('preprocessed')[0]
# 만약 dataset이 비었을 때 preprocessed를 활용
# 없으면 에러 발생 ㄱ

data_path = os.path.join("./dataset/", dataset)
meta_name = dataset + "_transcript.txt"
textgrid_name = "textgrids.zip"

### set GPU number ###
train_visible_devices = "0,1"
synth_visible_devices = "1"

# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
### kss ###
# !! aihub 기준 16000이라 수정
sampling_rate = 16000
filter_length = 1024
hop_length = 256
win_length = 1024

### kss ###
max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0
mel_fmax = 8000

f0_min = 71.0
f0_max = 792.8
energy_min = 0.0
energy_max = 283.72


# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000

# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = "./results"


# Optimizer
batch_size = 16
epochs = 10000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'hifigan'
vocoder_pretrained_model_name = dataset + "_g_00215000.pt"
vocoder_pretrained_model_path = os.path.join("./vocoder/pretrained_models/", vocoder_pretrained_model_name)


# HiFi-GAN parameter
resblock_kernel_sizes = [3,7,11]
upsample_rates = [8,8,2,2]
resblock = "1"
upsample_rates = [8,8,2,2]
upsample_kernel_sizes = [16,16,4,4]
upsample_initial_channel = 128
resblock_kernel_sizes = [3,7,11]
resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
# 실수로 컴마까지 다 가져와버림


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
eval_step = 5000
eval_size = 256
log_step = 1000
clear_Time = 20


