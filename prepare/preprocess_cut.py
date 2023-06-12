import os

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub import effects

rootPath = "data_svc/raw"
outPath = "data_svc/waves"
os.makedirs(outPath, exist_ok=True)

chunk_idx = 1
for file in os.listdir(f"./{rootPath}"):
    if (file.endswith(".wav")):
        audio = AudioSegment.from_wav(
            f"./{rootPath}/{file}").set_channels(1).set_frame_rate(16000)
        audio_norm = effects.normalize(audio, 6)  # max - 6dB
        audio_chunks = split_on_silence(
            audio_norm,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=300,
        )
        for chunk in audio_chunks:
            chunk = effects.normalize(chunk, 6)
            chunk.export(os.path.join(
                outPath, f"{chunk_idx:04d}.wav"), format="wav")
            chunk_idx = chunk_idx+1
