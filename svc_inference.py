import os
import torch
import librosa
import pyworld
import argparse
import numpy as np

from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["model_g"])
    return model


def compute_f0(path):
    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
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


ppg_path = "uni_svc_tmp.ppg.npy"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = Generator(hp)
    load_svc_model(args.model, model)

    os.system(f"python svc_inference_ppg.py -w {args.wave} -p {ppg_path}")

    ppg = np.load(ppg_path)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0(args.wave)
    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]

    model.eval(inference=True)
    model.to(device)
    with torch.no_grad():
        ppg = ppg.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        audio = model.inference(ppg, pit)
        audio = audio.cpu().detach().numpy()

    write("uni_svc_out.wav", hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-i', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    args = parser.parse_args()

    main(args)
