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
