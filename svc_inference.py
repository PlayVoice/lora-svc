import os
import torch
import librosa
import pyworld
import argparse
import numpy as np

from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator
from effects.equalizer import Equalizer


def after_eq(x):
    y = np.zeros(len(x))
    eq_gains = [-30, -10, 10, 10, 5, 0, -5, -10, -10, -10]
    eq = Equalizer(eq_gains, sample_rate=16000)
    eq.dump()
    # Start Processing
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = eq.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


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


def compute_f0_nn(filename, device):
    import torchcrepe
    # Load audio
    audio, sr = torchcrepe.load.audio(filename)
    # Here we'll use a 10 millisecond hop length
    hop_length = int(sr / 100.0)
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "tiny"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    # CREPE was not trained on silent audio. some error on silent need filter.
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 9)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch


ppg_path = "uni_svc_tmp.ppg.npy"


def main(args):
    os.system(f"python svc_inference_ppg.py -w {args.wave} -p {ppg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = Generator(hp)
    load_svc_model(args.model, model)

    ppg = np.load(ppg_path)
    pos = [1, 2]
    pos = np.tile(pos, ppg.shape[0])
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0_nn(args.wave, device)
    pit = torch.FloatTensor(pit)
    pos = torch.LongTensor(pos)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]
    pos = pos[:len_min]

    model.eval(inference=True)
    model.to(device)
    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        ppg = ppg.unsqueeze(0).to(device)
        pos = pos.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        audio = model.inference(spk, ppg, pos, pit)
        audio = audio.cpu().detach().numpy()

    audio = after_eq(audio)
    write("uni_svc_out.wav", hp.audio.sampling_rate, audio)


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
