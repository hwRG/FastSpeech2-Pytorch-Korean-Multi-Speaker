import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths, Embedding, SpeakerIntegrator
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    # !! embedding을 위한 파라미터 추가
    def __init__(self, n_speakers=1, speaker_embed_dim=256, speaker_embed_std=0.01, use_postnet=True):
        super(FastSpeech2, self).__init__()

        # !! speaker embedding 추가
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim
        self.speaker_embed_std = speaker_embed_std

        self.speaker_embeds = Embedding(n_speakers, speaker_embed_dim, padding_idx=None, std=speaker_embed_std)

        self.encoder = Encoder()

        # !! Speaker를 통합하는 레이어
        self.speaker_integrator = SpeakerIntegrator()

        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    # !! speaker_ids 추가
    def forward(self, src_seq, src_len, speaker_ids, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)


        """
        # !! 나뉘어진 mel_len 텐서를 다시 한 묶음으로 만들어냄
        # 파라미터를 넣지 않아서 발생한 문제였음
        mels_list = []
        for mels in mel_len:
            mel_tensor = 0
            for mel in mels:
                mel_tensor += int(mel)
            mels_list.append(mel_tensor)
        mels_list = torch.tensor(mels_list).to(device)
        """
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        
        # !! 스피커 임베딩 레이어 추가
        # !!! speakers_ids가 n_spkers와 비슷해야 하는건가?
        #speaker_embed = self.speaker_embeds(speaker_ids)

        # 해당 에러는 input size가 다를 때 나오는 에러임
        encoder_output = self.encoder(src_seq, src_mask)
        
        # !! encoder output에 하나로 뭉친 output 추가
        #encoder_output = self.speaker_integrator(encoder_output, speaker_embed)

        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                    encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)
        
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
