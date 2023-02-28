import os
import logging
import numpy as np
import librosa
import pyworld
import utils


def compute_f0(path):
    x, sr = librosa.load(path, fs=16000)
    assert sr == 16000
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=900,
        frame_period=1000 * 16000 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0


if __name__ == "__main__":

    files = open("./filelists/train.txt", "w", encoding="utf-8")

    rootPath = "../data_opencpop/waves/"
    outPath = "../data_opencpop/pitch/"

    for file in os.listdir(f"./{rootPath}"):
        if file.endswith(".wav"):
            file = file[:-4]
            wav_path = f"./{rootPath}/{file}.wav"
            featur_pit = compute_f0(wav_path)

            np.save(
                f"{outPath}/{file}.nsf",
                featur_pit,
                allow_pickle=False,
            )

            path_wave = f"../data_opencpop/waves/{file}.wav"
            path_pitch = f"../data_opencpop/pitch/{file}.nsf"
            path_whisper = f"../data_opencpop/whisper/{file}.ppg"
            print(
                f"{path_wave}|{path_pitch}|{path_whisper}",
                file=files,
            )

    files.close()
