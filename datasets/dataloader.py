import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import DistributedSampler, DataLoader, Dataset
from collections import Counter

from utils.utils import read_wav_np
from utils.stft import TacotronSTFT


def create_dataloader(hp, args, train, device):
    if train:
        dataset = MelFromDisk(hp, hp.data.train_dir, hp.data.train_meta, args, train, device)
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=False,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)

    else:
        dataset = MelFromDisk(hp, hp.data.val_dir, hp.data.val_meta, args, train, device)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, data_dir, metadata_path, args, train, device):
        random.seed(hp.train.seed)
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = data_dir
        metadata_path = os.path.join(data_dir, metadata_path)
        self.meta = self.load_metadata(metadata_path)
        self.stft = TacotronSTFT(hp.audio.filter_length, hp.audio.hop_length, hp.audio.win_length,
                                 hp.audio.n_mel_channels, hp.audio.sampling_rate,
                                 hp.audio.mel_fmin, hp.audio.mel_fmax, center=False, device=device)

        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.shuffle = hp.train.spk_balanced

        if train and hp.train.spk_balanced:
            # balanced sampling for each speaker
            speaker_counter = Counter((spk_id \
                                       for audiopath, text, spk_id in self.meta))
            weights = [1.0 / speaker_counter[spk_id] \
                       for audiopath, text, spk_id in self.meta]

            self.mapping_weights = torch.DoubleTensor(weights)

        elif train:
            weights = [1.0 / len(self.meta) for _, _, _ in self.meta]
            self.mapping_weights = torch.DoubleTensor(weights)


    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = torch.multinomial(self.mapping_weights, 1).item()
            return self.my_getitem(idx)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping_weights)

    def my_getitem(self, idx):
        wavpath, _, _ = self.meta[idx]
        wavpath = os.path.join(self.data_dir, wavpath)
        sr, audio = read_wav_np(wavpath)

        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = self.get_mel(wavpath)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length -1
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio_len = self.hp.audio.segment_length
            audio = audio[:, audio_start:audio_start + audio_len]

        return mel, audio

    def get_mel(self, wavpath):
        melpath = wavpath.replace('.wav', '.mel')
        try:
            mel = torch.load(melpath, map_location='cpu')
            assert mel.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, mel.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            mel = self.stft.mel_spectrogram(wav)

            mel = mel.squeeze(0)

            torch.save(mel, melpath)

        return mel

    def load_metadata(self, path, split="|"):
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip().split(split)
                metadata.append(stripped)

        return metadata