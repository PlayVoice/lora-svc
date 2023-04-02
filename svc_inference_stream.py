import os
import torch
import librosa
import pyworld
import argparse
import numpy as np
import datetime
import random

from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator

import os
import numpy as np
import argparse
import torch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, log_mel_spectrogram


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_whisper(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def pred_ppg(whisper: Whisper, audio):
    audln = audio.shape[0]
    ppgln = audln // 320
    mel = log_mel_spectrogram(audio).to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(
            0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        return ppg


def compute_f0(x, sr=16000):
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=900,
        frame_period=1000 * 160 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs=16000)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0


def load_maxgan(checkpoint_path, config_path):
    conf = OmegaConf.load(config_path)
    maxgan = Generator(conf)
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    maxgan.load_state_dict(checkpoint_dict["model_g"])
    return maxgan


def main(args):
    setup_seed(1234)

    whisper = load_whisper(os.path.join("whisper_pretrain", "medium.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maxgan = load_maxgan(args.model, args.config)
    maxgan.eval(inference=True)
    maxgan.to(device)

    speaker = np.load(args.spk)
    speaker = torch.FloatTensor(speaker)
    speaker = speaker.unsqueeze(0).to(device)

    x, sr = librosa.load(args.wave, sr=16000)
    assert sr == 16000
    len_x = x.shape[0]

    # can not change these parameters
    hop_length = 320  # whisper hop size
    hop_count = len_x // hop_length
    hop_frame = 10
    hop_sample = hop_frame * hop_length
    stream_chunk = 50
    stream_index = 0
    stream_out_wav = []

    while (stream_index + stream_chunk < hop_count):
        if (stream_index == 0):  # start frame
            cut_s = stream_index
            cut_s_wav = 0
        else:
            cut_s = stream_index - hop_frame
            cut_s_wav = hop_sample

        if (stream_index + stream_chunk > hop_count - hop_frame):  # end frame
            cut_e = stream_index + stream_chunk
            cut_e_wav = 0
        else:
            cut_e = stream_index + stream_chunk + hop_frame
            cut_e_wav = -1 * hop_sample

        x_chunk = x[cut_s * hop_length:cut_e * hop_length]
        p_chunk = pred_ppg(whisper, x_chunk)
        postion = [1, 2]
        postion = np.tile(postion, p_chunk.shape[0])
        p_chunk = np.repeat(p_chunk, 2, 0)  # 320 PPG -> 160 * 2
        f_chunk = compute_f0(x_chunk)
        p_chunk = torch.FloatTensor(p_chunk)
        f_chunk = torch.FloatTensor(f_chunk) * 1.5
        postion = torch.LongTensor(postion)
        len_ppg = p_chunk.size()[0]
        len_pit = f_chunk.size()[0]
        len_min = min(len_ppg, len_pit)
        p_chunk = p_chunk[:len_min]
        f_chunk = f_chunk[:len_min]
        postion = postion[:len_min]
        with torch.no_grad():
            p_chunk = p_chunk.unsqueeze(0).to(device)
            postion = postion.unsqueeze(0).to(device)
            f_chunk = f_chunk.unsqueeze(0).to(device)
            audio = maxgan.inference(speaker, p_chunk, postion, f_chunk)
            o_chunk = audio.cpu().detach().numpy()
        o_chunk = o_chunk[cut_s_wav:cut_e_wav]
        stream_out_wav.extend(o_chunk)
        stream_index = stream_index + stream_chunk
        print(datetime.datetime.now())

    if (stream_index < hop_count):
        cut_s = stream_index - hop_frame
        cut_s_wav = hop_sample
        x_chunk = x[cut_s * hop_length:]
        p_chunk = pred_ppg(whisper, x_chunk)
        postion = [1, 2]
        postion = np.tile(postion, p_chunk.shape[0])
        p_chunk = np.repeat(p_chunk, 2, 0)  # 320 PPG -> 160 * 2
        f_chunk = compute_f0(x_chunk)
        p_chunk = torch.FloatTensor(p_chunk)
        f_chunk = torch.FloatTensor(f_chunk) * 1.5
        postion = torch.LongTensor(postion)
        len_ppg = p_chunk.size()[0]
        len_pit = f_chunk.size()[0]
        len_min = min(len_ppg, len_pit)
        p_chunk = p_chunk[:len_min]
        f_chunk = f_chunk[:len_min]
        postion = postion[:len_min]
        with torch.no_grad():
            p_chunk = p_chunk.unsqueeze(0).to(device)
            postion = postion.unsqueeze(0).to(device)
            f_chunk = f_chunk.unsqueeze(0).to(device)
            audio = maxgan.inference(speaker, p_chunk, postion, f_chunk)
            o_chunk = audio.cpu().detach().numpy()
        o_chunk = o_chunk[cut_s_wav:]
        stream_out_wav.extend(o_chunk)

    stream_out_wav = np.asarray(stream_out_wav)
    write("svc_out.wav", 16000, stream_out_wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    args = parser.parse_args()

    main(args)
