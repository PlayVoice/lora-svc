import os
import argparse
import librosa
import torch
from scipy.io.wavfile import write
import numpy as np

SAMPLE_RATE = 48000
frame_length=480000
hop_length=480000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    audio, sr = librosa.load(args.wave, sr=16000)
    model = torch.jit.load(os.path.join(
        "bandex", "hifi-gan-bwe-vctk-48kHz.pt")).to(device)
    audio_length=len(audio)
    pad_length = frame_length - (audio_length - frame_length) % hop_length # calculate the padding length
    audio=np.pad(audio, (0, pad_length), mode='constant') # pad the array with zeros
    # split the audio into frames of 30 seconds with zero overlap
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frames = np.transpose(frames, (1, 0))
    # initialize an empty list to store the processed frames
    bwe_frames = []
    with torch.no_grad():
        # loop over the frames
        for idx,frame in enumerate(frames):
            # convert the frame to a tensor and send it to the device
            frame_gpu = torch.from_numpy(np.copy(frame)).to(device)
            # process the frame with the model
            bwe_frame = model(frame_gpu, sr).data.cpu().float().numpy()
            # append the processed frame to the list
            if idx==len(frames)-1:
                bwe_frames.append(bwe_frame[:-pad_length*int(48000/16000)])
            else:
                bwe_frames.append(bwe_frame)
    # concatenate the processed frames into a single array
    bwe_audio = np.concatenate(bwe_frames)
    write("svc_out_48k.wav", SAMPLE_RATE, bwe_audio)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    args = parser.parse_args()
    main(args)
