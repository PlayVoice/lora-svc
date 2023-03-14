import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from torch.nn import Conv1d

from .lvcnet import LVCBlock
from .nsf import SourceModuleHnNSF

MAX_WAV_VALUE = 32768.0

class SpeakerAdapter(nn.Module):

    def __init__(self,
                speaker_dim,
                adapter_dim,
                epsilon=1e-5
                ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        y = y.transpose(1, -1)
        return y

class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = hp.audio.n_mel_channels
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size

        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = nn.ModuleList()

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(hp.gen.strides))
        self.m_source = SourceModuleHnNSF()
        self.noise_convs = nn.ModuleList()

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for i, stride in enumerate(hp.gen.strides):
            # spk
            self.adapter.append(SpeakerAdapter(256, channel_size))
            # net
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.audio.n_mel_channels,
                    stride=stride,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )
            # nsf
            if i + 1 < len(hp.gen.strides):
                stride_f0 = np.prod(hp.gen.strides[i + 1:])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        channel_size,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, channel_size, kernel_size=1))

        # 1024 should change by your whisper out
        self.cond_pre = nn.Linear(1024, self.mel_channel)
        self.cond_pos = nn.Embedding(3, self.mel_channel)
    
        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(hp.gen.noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, spk, c, pos, f0, z):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        c = c + torch.randn_like(c)
        c = self.cond_pre(c)                # [B, L, D]
        p = self.cond_pos(pos)
        c = c + p
        c = torch.transpose(c, 1, -1)       # [B, D, L]
        z = self.conv_pre(z)                # (B, c_g, L)

        for i, res_block in enumerate(self.res_stack):
            res_block.to(z.device)
            x_source = self.noise_convs[i](har_source)
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)
            z = self.adapter[i](z, spk)     # adapter
            z = z + x_source

        z = self.conv_post(z)               # (B, 1, L * 160)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, spk, ppg, pos, f0, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        # zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        # mel = torch.cat((c, zero), dim=2)
        if z is None:
            z = torch.randn(1, self.noise_dim, ppg.size(1)).to(ppg.device)
        audio = self.forward(spk, ppg, pos, f0, z)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio

if __name__ == '__main__':
    hp = OmegaConf.load('../config/default.yaml')
    model = Generator(hp)

    c = torch.randn(3, 10, 1024)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 1600])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
