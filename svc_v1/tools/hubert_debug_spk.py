import os
import torch
import argparse
import numpy as np
import librosa
import pyworld

import utils

from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
from singer_vc.models import SynthesizerTrn

from scipy.io import wavfile


# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def compute_f0(filename, fs=16000, hop=320):
    x, sr = librosa.load(filename, fs)
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * hop / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0[:-1]  # f0_len = mel_len + 1


def coarse_f0(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1
    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "please enter embed parameter ..."
    parser.add_argument("-s", "--source", help="input wave", dest="source")
    parser.add_argument("-hu", "--hubert", help="input hubert", dest="hubert")
    parser.add_argument("-te", "--target_embed", help="256维声纹", dest="target_embed")

    args = parser.parse_args()
    source_file = args.source
    hubert_file = args.hubert
    print("source file is :", source_file)
    print("hubert file is :", hubert_file)

    target_embed_file = args.target_embed
    print("target embed is :", target_embed_file)

    hps = utils.get_hparams_from_file("./configs/singing_base.json")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda()

    # _ = utils.load_checkpoint("./logs/singer_vits_mel/G_230000.pth", net_g, None)
    net_g = torch.load("model_net_g.pth")
    _ = net_g.eval()

    nsff0 = compute_f0(source_file, hps.data.sampling_rate, hps.data.hop_length)
    pitch = coarse_f0(nsff0)

    # EMBEDDING
    target_embed = np.load(target_embed_file)
    target_embed = target_embed.astype(np.float32)
    # print(f"target_embed={target_embed}")

    # VC
    sid_tgt = torch.FloatTensor(target_embed).cuda().unsqueeze(0)

    phone = np.load(hubert_file)
    n_num = len(phone)
    phone = phone[:n_num]
    pitch = pitch[:n_num]
    nsff0 = nsff0[:n_num]
    phone = torch.LongTensor(phone)
    pitch = torch.LongTensor(pitch)
    nsff0 = torch.FloatTensor(nsff0)
    phone_lengths = phone.size()[0]

    with torch.no_grad():
        phone = phone.cuda().unsqueeze(0)
        pitch = pitch.cuda().unsqueeze(0)
        nsff0 = nsff0.cuda().unsqueeze(0)
        phone_lengths = torch.LongTensor([phone_lengths]).cuda()
        audio = (
            net_g.infer(phone, phone_lengths, pitch, nsff0, sid_tgt)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    os.system(f"cp {source_file} vc_in.wav")

    save_wav(audio, "hubert_out.wav", hps.data.sampling_rate)
