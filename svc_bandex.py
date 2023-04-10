import os
import argparse
import librosa
import torch
from scipy.io.wavfile import write

SAMPLE_RATE = 48000


def main(args):
    audio, sr = librosa.load(args.wave, sr=16000)
    model = torch.jit.load(os.path.join(
        "bandex", "hifi-gan-bwe-vctk-48kHz.pt"))
    audio = torch.from_numpy(audio)
    with torch.no_grad():
        bwe_audio = model(audio, sr).data.cpu().float().numpy()
    write("svc_out_48k.wav", SAMPLE_RATE, bwe_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    args = parser.parse_args()
    main(args)
