import os
import torch
import librosa
import argparse
import numpy as np
import torchcrepe

from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["model_g"])
    return model


def compute_f0_nn(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
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
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 9)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python svc_inference_ppg.py -w {args.wave} -p {args.ppg}")
    os.system(f"python svc_inference_ppg.py -w {args.wave} -p {args.ppg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = Generator(hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    pos = [1, 2, 3, 4, 5, 6]
    pos = np.tile(pos, ppg.shape[0])
    ppg = np.repeat(ppg, 6, 0)  # 20ms:16k:320 -> 20ms:48k:960 960/160=6
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0_nn(args.wave, device)
    pit = np.repeat(pit, 3, 0) # 10ms:16k:160 -> 10ms:48k:480 480/160=3
    if (args.statics == None):
        print("don't use pitch shift")
    else:
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        singer_ave, singer_min, singer_max = np.load(args.statics)
        print(f"singer pitch statics: mean={singer_ave:0.1f}, \
                min={singer_min:0.1f}, max={singer_max:0.1f}")

        shift = np.log2(singer_ave/source_ave) * 12
        if (singer_ave >= source_ave):
            shift = np.floor(shift)
        else:
            shift = np.ceil(shift)
        shift = 2 ** (shift / 12)
        pit = pit * shift

    pit = torch.FloatTensor(pit)
    pos = torch.LongTensor(pos)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]
    pos = pos[:len_min]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        ppg = ppg.unsqueeze(0).to(device)
        pos = pos.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        audio = model.inference(spk, ppg, pos, pit)
        audio = audio.cpu().detach().numpy()

        pitwav = model.pitch2wav(pit)
        pitwav = pitwav.cpu().detach().numpy()

    write("svc_out.wav", hp.audio.sampling_rate, audio)
    write("svc_out_pitch.wav", hp.audio.sampling_rate, pitwav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('-p', '--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('-t', '--statics', type=str,
                        help="Path of pitch statics.")
    args = parser.parse_args()

    main(args)
