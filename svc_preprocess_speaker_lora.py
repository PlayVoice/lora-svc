import os
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("dataset_path", type=str,
                        help="Path to dataset waves.")
    data_svc = parser.parse_args().dataset_path

    if os.path.isdir(os.path.join(data_svc, "speaker")):
        subfile_num = 0
        speaker_ave = 0
        for file in os.listdir(os.path.join(data_svc, "speaker")):
            if file.endswith(".npy"):
                source_embed = np.load(os.path.join(data_svc, "speaker", file))
                source_embed = source_embed.astype(np.float32)
                speaker_ave = speaker_ave + source_embed
                subfile_num = subfile_num + 1
        speaker_ave = speaker_ave / subfile_num
        print(speaker_ave)
        np.save(os.path.join(data_svc, "lora_speaker.npy"),
                speaker_ave, allow_pickle=False)

    if os.path.isdir(os.path.join(data_svc, "pitch")):
        subfile_num = 0
        speaker_ave = 0
        speaker_max = 0
        speaker_min = 1000
        for file in os.listdir(os.path.join(data_svc, "pitch")):
            if file.endswith(".npy"):
                pitch = np.load(os.path.join(data_svc, "pitch", file))
                pitch = pitch.astype(np.float32)
                pitch = pitch[pitch > 0]
                speaker_ave = speaker_ave + pitch.mean()
                subfile_num = subfile_num + 1
                if (speaker_max < pitch.max()):
                    speaker_max = pitch.max()
                    print(f'{file} has {speaker_max}')
                if (speaker_min > pitch.min()):
                    speaker_min = pitch.min()
                    print(f'{file} has {speaker_min}')
        speaker_ave = speaker_ave / subfile_num
        pitch_statics = [speaker_ave, speaker_min, speaker_max]
        print(pitch_statics)
        np.save(os.path.join(data_svc, "lora_pitch_statics.npy"),
                pitch_statics, allow_pickle=False)
