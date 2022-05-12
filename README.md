# FastSpeech2-Pytorch-Korean-Multi-Speaker

FastSpeech2에 HiFi-GAN Vocoder를 결합하여, 한국어 Multi-Speaker TTS로 구현한 프로젝트 입니다. 



## Base on

본 프로젝트는 [한동대 DLLAB](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)에서 구현한 한국어 데이터셋 [KSS](http://kaggle.com/bryanpark/korean-single-speaker-speech-dataset)에서 동작하는 FastSpeech2 소스 코드를 기반으로 구현했습니다.





## Introduction

- 본 주제는 [배리어프리 앱개발 콘테스트](https://www.autoeverapp.kr/)에 참여하는 **‘보이는 개인화 AI 스피커’ 프로젝트의 TTS 개발**을 목표로 진행됩니다. 기존에 식상한 '시리', '빅스비', '아리'의 목소리가 아닌 사용자가 원하는 주변 사람의 목소리로 대체합니다. (ex. 배우자, 아들, 딸, 부모님 등)

- AI 스피커라는 즉각적인 생성에 대응하기 위해 기존에 뛰어난 성능의 Tacotron2와 Waveglow 대신 **Non-Autoregressive Acoustic Model FastSpeech2**과 **GAN 기반 Vocoder Model HiFi-GAN**을 채택하여 퀄리티와 생성 속도 모두 고려할 수 있도록 합니다.





## Project Purpose

프로젝트의 목표는 다음과 같습니다.

1. 합성 속도와 퀄리티를 위해 Acoustic-Fastspeech2, Vocoder-HiFiGAN 모델 활용
2. 소량의 데이터에 Adapt하기 위해 Transfer Learning 활용
3. Pre-train을 위한 Multi-Speaker 데이터를 학습하기 위해 Speaker Embedding 구현 
4. 한국어 데이터셋에 학습 전 과정이 end-to-end로 수행될 수 있도록 파이프라인 구성





## Dataset

- 결과 확인을 위해, AIHub의 [자유대화 음성](https://aihub.or.kr/aidata/30703)을 활용해 Multi-Speaker Pre-training을 수행합니다. 
- 데이터, 스피커 명이 숫자 6자리가 포함되도록 수정합니다. 
- ~~제시된 스크립트를 통해 약 45분 분량의 음성 데이터를 메뉴얼에 따라 녹음합니다. (준비 중)~~





## Add from Previous Project

활용한 코드에서 추가된 내용은 다음과 같습니다.

1. Speaker Embedding 구현 

   + 모델에 Embedding layer를 추가
   + Encoder output과 더하는 코드 구현 (Embedding, SpeakerIntegrator)
   + Embedding 정보를 가져오고 저장하는 get_speakers() 함수 구현

2. data_preprocessing.py - 아래 항목을 모두 포함하는 end-to-end 데이터 전처리 구현

   ![data_preprocessing](/asset/data_preprocessing.png)

3. 긴 문장에 대한 불안정한 합성 시 대응

   - 특수문자 단위로(문장 단위) 끊어 합성 후 이어 붙이도록 설정

   ![cut_and_synthesize](/asset/cut_and_synthesize.png)

4. G2pk 소스 코드를 불러와 숫자, 영어만 변환하도록 적용

   - 기존 [G2pk](https://github.com/Kyubyong/g2pK)의 패키지를 pip 설치 없이 숫자, 영어만 한글로 변환 하도록 수정





## How to Train

- FastSpeech2는 학습 전에, Kaldi로 구현된 [Montral Forced Alinger](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html)로 생성된 **align**이 요구됩니다.
- 학습 과정 중 evaluate을 위해 [HiFi-GAN](https://github.com/hwRG/HiFi-GAN-Pytorch)으로 학습한 generator를 vocoder/pretrained_models에 포함되어야 합니다.

1. 개인 데이터에 KSS의 transcript 형식에 따라 생성하거나, 형식에 맞게 data_preprocessing.py의 json 불러오는 함수를 커스텀하여 transcript를 생성합니다.
2. 생성된 transcript와 데이터의 디렉토리를 dataset에 보관 후 data_preprocessing.py를 실행합니다.  
3. MFA 작업이 완료되고 Textgrid가 최상위 디렉토리에 생성된 것을 확인합니다.
4. preprocess.py를 수행하고 preprocessed 폴더에 전처리된 데이터가 생성된 것을 확인합니다.
5. hparam.py의 batch size, HiFi-GAN generator의 path 등 설정 후 train.py를 실행합니다.
   - 이후 재 학습을 하게 될 경우 train.py --restore_step [step수]로 재학습이 가능합니다.





## How to Transfer Learning

- Multi-Speaker에 대한 Pre-train일 경우 Pre-train 학습 시 자동으로 생성된 speaker_info.json을 준비합니다.
- speaker_info.json와 생성된 pth.tar 체크포인트를 ckpt에 위치하고 train.py --restore_step으로 학습합니다.





## How to Synthesize

- python synthesize.py --step [step수]를 통해 학습을 수행합니다.
  - 임의로 제시한 대본으로 합성 1, 2, 3번 선택
  - 직접 작성한 대본 생성은 4번 선택





## Model Pipeline

본 파이프라인은 서비스에 해당되는 TTS 학습 및 생성에 대한 flow 파이프라인 입니다.

![Transfer_Learning_Pipeline](/asset/Transfer_Learning_Pipeline.png)

- 컨테이너는 크게 4개로 분류됩니다. 
  1. 데이터의 path와 유저 정보 등을 담고 있는 데이터베이스 컨테이너
  2. Transcript 생성, 파일명 간소화, MFA로 Textgrid 추출, 모델에 필요한 Data Preprocessing 컨테이너
  3. Pre-Training을 위한 학습 컨테이너
  4. 새로운 데이터에 Fine-Tuning을 위한 학습 컨테이너

- 실제 서비스 상황엔 Pre-trianing 컨테이너 외 3개 컨테이너만 작동하게 됩니다.