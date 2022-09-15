import os
import utils
import torch
import argparse
import numpy as np

from time import *
from models import SynthesizerTrn
from prepare.preprocess_wave import FeatureInput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "please enter embed parameter ..."
    parser.add_argument("-s", "--wave", help="input wave", dest="source")
    parser.add_argument("-p", "--ppgs", help="input ppgs", dest="ppgs")
    parser.add_argument("-e", "--embe", help="input embe", dest="embed")
    input_args = parser.parse_args()

    hps = utils.get_hparams_from_file("./configs/singing_base.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )

    _ = utils.load_checkpoint("./logs/singing_base/G_200000.pth", net_g, None)
    _ = net_g.eval().to(dev)

    speaker = np.load(input_args.embed)
    phone = np.load(input_args.ppgs)

    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)
    pitch = featureInput.compute_f0(input_args.source)
    pitch = featureInput.coarse_f0(pitch)

    p_len = len(phone)
    phone = phone[:p_len]
    pitch = pitch[:p_len]

    p_len = torch.LongTensor([p_len]).to(dev)
    phone = torch.LongTensor(phone).unsqueeze(0).to(dev)
    pitch = torch.LongTensor(pitch).unsqueeze(0).to(dev)
    speaker = torch.FloatTensor(speaker).unsqueeze(0).to(dev)
    with torch.no_grad():
        audio = (
            net_g.infer(phone, p_len, pitch, None, speaker)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    featureInput.save_wav(audio, "vi-svc_out.wav")
