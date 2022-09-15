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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "please enter embed parameter ..."
    parser.add_argument("-s", "--source", help="input wave", dest="source")
    parser.add_argument("-se", "--source_embed", help="256维声纹", dest="source_embed")
    parser.add_argument("-te", "--target_embed", help="256维声纹", dest="target_embed")

    args = parser.parse_args()
    source_file = args.source
    print("source file is :", source_file)

    source_embed_file = args.source_embed
    target_embed_file = args.target_embed
    print("source embed is :", source_embed_file)
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

    nsf_f0 = compute_f0(source_file, hps.data.sampling_rate, hps.data.hop_length)
    audio, sampling_rate = load_wav_to_torch(source_file)

    # float
    y = audio / hps.data.max_wav_value
    y = y.unsqueeze(0)

    # EMBEDDING
    source_embed = np.load(source_embed_file)
    source_embed = source_embed.astype(np.float32)
    # print(f"source_embed={source_embed}")
    target_embed = np.load(target_embed_file)
    target_embed = target_embed.astype(np.float32)
    # print(f"target_embed={target_embed}")

    # VC
    sid_src = torch.FloatTensor(source_embed).cuda().unsqueeze(0)
    sid_tgt = torch.FloatTensor(target_embed).cuda().unsqueeze(0)

    spec = spectrogram_torch(
        y,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    ).cuda()
    spec_lengths = torch.LongTensor([spec.size(-1)]).cuda()
    nsff0 = torch.FloatTensor(nsf_f0)
    nsff0 = nsff0.cuda().unsqueeze(0)

    with torch.no_grad():
        audio_vc = (
            net_g.voice_conversion(
                spec, spec_lengths, nsff0, emb_src=sid_src, emb_tgt=sid_tgt
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    os.system(f"cp {source_file} vc_in.wav")

    save_wav(audio_vc, "vc_out.wav", hps.data.sampling_rate)
