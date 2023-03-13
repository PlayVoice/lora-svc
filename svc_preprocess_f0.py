import os
import logging
import numpy as np
import librosa
import pyworld
import utils


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


if __name__ == "__main__":

    files = open("./filelists/train.txt", "w", encoding="utf-8")

    rootPath = "./data_svc/waves/"
    outPath = "./data_svc/pitch/"
    os.makedirs(outPath, exist_ok=True)

    for spks in os.listdir(f"./{rootPath}"):
        if os.path.isdir(f"./{rootPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{rootPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    wav_path = f"./{rootPath}/{spks}/{file}.wav"
                    featur_pit = compute_f0(wav_path)

                    np.save(
                        f"{outPath}/{spks}/{file}.nsf",
                        featur_pit,
                        allow_pickle=False,
                    )

                    path_spk = f"./data_svc/ids/{spks}.npy"
                    path_wave = f"./data_svc/waves/{spks}/{file}.wav"
                    path_pitch = f"./data_svc/pitch/{spks}/{file}.nsf.npy"
                    path_whisper = f"./data_svc/whisper/{spks}/{file}.ppg.npy"
                    print(
                        f"{path_wave}|{path_pitch}|{path_whisper}|{path_spk}",
                        file=files,
                    )

    files.close()
