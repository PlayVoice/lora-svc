import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import joblib
import argparse


from fairseq import checkpoint_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/TencentGameMate/chinese_speech_pretrain
model_path = "./chinese-hubert-base.pt"
kmeans_model_path = "./hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl"  # layer 9

# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
model = model.half()
model.eval()

# K-means model
kmeans_model = joblib.load(open(kmeans_model_path, "rb"))
kmeans_model.verbose = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "please enter embed parameter ..."
    parser.add_argument("-s", "--wave", help="input wave", dest="source")
    input_args = parser.parse_args()
    wav_path = input_args.source
    if wav_path.endswith(".wav"):

        out_path = f"{wav_path[:-4]}_hubert.npy"

        feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.half().to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": 9,  # layer 9
        }

        with torch.no_grad():
            feats, _ = model.extract_features(**inputs)

        feats = feats.squeeze(0).float().cpu().numpy()
        pred = kmeans_model.predict(feats)
        pred = np.repeat(pred, 2)  # 20ms -> 10ms

        np.save(out_path, pred, allow_pickle=False)
