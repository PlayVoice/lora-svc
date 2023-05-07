import os
import argparse
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import json
import torch


def split(sound):
    """
    Split the given audio segment object into multiple segments based on silent parts.

    :param sound: AudioSegment object to split.
    :type sound: AudioSegment
    :return: A list of AudioSegment objects after splitting silence from the input audio.
    :rtype: List[AudioSegment]
    """

    # Calculate dBFS value of input audio for silence detection
    dBFS = sound.dBFS

    # Split the input audio segment based on silence using the specified parameters
    chunks = split_on_silence(sound,
                              min_silence_len=100,
                              silence_thresh=dBFS - 16,
                              keep_silence=100)

    # Return a list of AudioSegment objects representing the resulting split audio clips
    return chunks

def combine(_src):
    """
    Merge all wav audio files from the specified directory to a single AudioSegment object.

    :param _src: Directory path containing '.wav' audio files to be merged.
    :type _src: str
    :return: A single AudioSegment object containing merged audio files.
    :rtype: AudioSegment
    """
    audio = AudioSegment.empty()
    for i, filename in enumerate(os.listdir(_src)):
        if filename.endswith('.wav'):
            filename = os.path.join(_src, filename)
            try:
                audio += AudioSegment.from_wav(filename)
            except:
                pass
    return audio


def save_chunks(chunks, directory, sr=16000, duration=5 * 1000, channels=1, format='wav', ext='.wav'):
    """
    Save the given list of AudioSegment objects as separate audio files in a specified directory.

    :param chunks: A list of AudioSegment objects to be saved.
    :type chunks: List[AudioSegment]
    :param directory: The directory path where the output audio files will be saved.
    :type directory: str
    :param sr: Sample rate (in Hz) for the output audio files. Default is 16000.
    :type sr: int
    :param duration: Desired duration (in milliseconds) for each output audio file. Default is 5000ms (5 seconds).
    :type duration: int
    :param channels: Number of audio channels for the output audio files. Default is 1.
    :type channels: int
    :param format: Output file format for the audio files. Default is 'wav'.
    :type format: str
    :param ext: File extension for the output audio files. Default is '.wav'.
    :type ext: str
    :return: None
    :rtype: None
    """

    # Create the target output directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize a counter variable for naming output files
    counter = 0

    # Define the desired length of each output chunk
    target_length = duration

    # Split the input audio into chunks of desired length
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # If the last output chunk is longer than the target length,
            # we start a new one
            output_chunks.append(chunk)
    
    # Save each output chunk as a separate audio file in the specified directory
    for chunk in output_chunks:
        chunk = chunk.set_frame_rate(sr)
        chunk = chunk.set_channels(channels)
        counter = counter + 1
        chunk.export(os.path.join(directory, str(counter) + ext), format=format)


def process(item):
    """
    Process a single item by combining, splitting and saving audio files in the specified directories.

    :param item: A tuple containing speaker directory path and arguments for processing.
    :type item: Tuple[str, Any]
    :return: None
    :rtype: None
    """
    # Extract speaker directory path and argument values from the input item tuple
    spkdir, args = item

    # Define input and output directories for processing
    out_dir = args.out_dir
    in_dir = args.in_dir
    
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    print(f"Combining audio files in '{in_dir}'...")
    # Combine all .wav files in the input directory into a single AudioSegment object
    sound = combine(in_dir)
    print("Audio files combined successfully.")

    print("Splitting audio into chunks based on silence...")
    # Split the AudioSegment object into multiple chunks based on silence
    chunks = split(sound)
    print("Audio split successfully.")

    print(f"Saving {len(chunks)} chunks to '{out_dir}'...")
    # Save each chunk as a separate .wav file in the output directory
    save_chunks(chunks, out_dir, sr=args.sr, duration=args.duration)
    print("Chunks saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (in Hz) of output audio files. Default is 16000.")
    parser.add_argument("--duration", type=int, default=10 * 1000, help="Desired duration (in milliseconds) of output audio files. Default is 10,000 ms (10 seconds).")
    parser.add_argument("--in_dir", type=str, default="./data_svc/waves-raw", help="Directory path containing input audio files.")
    parser.add_argument("--out_dir", type=str, default="./data_svc/waves-16k", help="Directory path where output audio files will be saved.")
    parser.add_argument("--top_db", type=int, default=25, help="Maximum dBFS level for detecting silent parts. Default is 25.")
    args = parser.parse_args()

    process((args.in_dir, args))