#!/usr/bin/env python3
"""
This module provides a graphical user interface for the SVC process.

The SVC process is a voice conversion system that uses a generative adversarial network (GAN) to convert the voice of a speaker to another speaker.

The module requires the following files and directories:

- whisper/inference.py: a script that extracts the content encoding from a wav file
- pitch/inference.py: a script that extracts the F0 parameter from a wav file
- svc_inference.py: a script that performs the voice conversion using a GAN model
- configs/maxgan.yaml: a configuration file for the GAN model
- maxgan_g.pth: a pretrained GAN model file
- lora-svc/data_svc/singer: a directory that contains the spk files for different singers

The module allows the user to select a wav file and a spk file, and run the SVC process to generate an output wav file with the converted voice.
"""

# import tkinter module and other modules
import tkinter as tk
from tkinter import filedialog
import os
import subprocess

# import sys and platform modules
import sys
import platform

# import logging module and configure logging level and format
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# import playsound module
from playsound import playsound

# define global variables for working directory, LD_LIBRARY_PATH, model file and checkpoint file 
WORKING_DIR = os.getcwd()
LD_LIBRARY_PATH = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
MODEL_FILE = "maxgan_g.pth"
CHECKPOINT_FILE = ""

# define a function that converts a file to wav format using ffmpeg
def convert_to_wav(file):
    """
    Converts a file to wav format using ffmpeg.

    Parameters:
    file (str): the path of the input file

    Returns:
    str: the path of the output wav file
    """
    # get the file name and extension
    file_name = os.path.basename(file)
    file_ext = os.path.splitext(file_name)[1]

    # check if the file extension is wav
    if file_ext == ".wav":
        # no need to convert, just return the original file name
        return file_name
    else:
        # use ffmpeg to convert the file to wav format with -y option
        new_file_name = os.path.splitext(file_name)[0] + ".wav"
        new_file_path = os.path.join(WORKING_DIR, new_file_name)
        subprocess.run(["ffmpeg", "-y", "-i", file, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", new_file_path])
        return new_file_path

# define a function that finds the highest numbered checkpoint file in a given directory
def find_highest_checkpoint(dir_name):
    """
    Finds the highest numbered checkpoint file in a given directory.

    Parameters:
    dir_name (str): the path of the directory

    Returns:
    str: the path of the highest checkpoint file
    """
    # initialize the highest number and file name variables
    highest_number = 0
    highest_file = ""

    # loop through all the files in the directory with pt extension using itertools
    for file in itertools.filterfalse(lambda x: not x.endswith(".pt"), os.listdir(dir_name)):
        # extract the number from the file name
        number = int(os.path.splitext(file)[0].split("_")[-1])

        # compare the number with the highest number so far
        if number > highest_number:
            # update the highest number and file name variables
            highest_number = number
            highest_file = os.path.join(dir_name, file)

    # return the highest file name
    return highest_file

# define a function that takes a wav file and a spk file as arguments and runs all the commands
def run_all(wav_file, spk_file):
    """
    Takes a wav file and a spk file as arguments and runs all the commands for the SVC process.

    Parameters:
    wav_file (str): the path of the input wav file
    spk_file (str): the path of the input spk file

    Returns:
    None
    """
    try:
        # set working directory and LD_LIBRARY_PATH variables using global variables defined earlier
        os.environ["PYTHONPATH"] = WORKING_DIR
        os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

        # use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage
        subprocess.run(["python3", "whisper/inference.py", "-w", wav_file, "-p", wav_file + ".ppg.npy"])

        # extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser
        subprocess.run(["python3", "pitch/inference.py", "-w", wav_file, "-p", wav_file + ".csv"])

        # specify parameters and infer using the maxgan_g.pth model and checkpoint file names defined earlier as global variables 
        subprocess.run(["python3", "svc_inference.py", "--config", "configs/maxgan.yaml", "--model", MODEL_FILE, "--spk", spk_file, "--wave", wav_file, "--ppg", wav_file + ".ppg.npy", "--pit", wav_file + ".csv"])

        # rename the output file to include the input file name and spk file name
        input_file_name = os.path.splitext(os.path.basename(wav_file))[0]
        spk_file_name = os.path.splitext(os.path.basename(spk_file))[0]
        output_file = f"svc_{input_file_name}_out_{spk_file_name}.wav"
        output_path = os.path.join(WORKING_DIR, output_file)
        os.rename(os.path.join(WORKING_DIR, "svc_out.wav"), output_path)

        # log the successful completion of the process
        logging.info(f"SVC process completed for {wav_file} and {spk_file}. Output file: {output_path}")

        # update the global variable for output file name
        global out_file
        out_file = output_path

        # enable the play button widget
        play_button.config(state="normal")
    except Exception as e:
        # log the exception and display an error message
        logging.error(f"SVC process failed for {wav_file} and {spk_file}. Exception: {e}")
        tk.messagebox.showerror("SVC Error", f"An error occurred while running the SVC process. Please check the log for details.")

# define a function that plays the output file using playsound
def play_output():
    """
    Plays the output file using playsound.

    Parameters:
    None

    Returns:
    None
    """
    try:
        # check if there is an output file name
        if out_file:
            # play the output file using playsound
            playsound(out_file)
            # log the successful playback of the output file
            logging.info(f"Played output file: {out_file}")
        else:
            # display an error message if there is no output file name
            tk.messagebox.showerror("SVC Error", f"No output file to play.")
    except Exception as e:
        # log the exception and display an error message
        logging.error(f"Failed to play output file: {out_file}. Exception: {e}")
        tk.messagebox.showerror("SVC Error", f"An error occurred while playing the output file. Please check the log for details.")

# create a tkinter window object 
window = tk.Tk()

# set window title and size 
window.title("SVC GUI")
window.geometry("600x400")

# create a label widget to display instructions
label = tk.Label(window, text="Select a wav file and a spk file to run the SVC process", font=("Arial", 16))
label.pack()

# create a button widget to select a wav file
wav_button = tk.Button(window, text="Select wav file", font=("Arial", 14), command=lambda: select_file("wav"))
wav_button.pack()

# create a button widget to select a spk file
spk_button = tk.Button(window, text="Select spk file", font=("Arial", 14), command=lambda: select_file("spk"))
spk_button.pack()

# create a button widget to run the SVC process
run_button = tk.Button(window, text="Run SVC", font=("Arial", 14), command=lambda: run_all(wav_file, spk_file))
run_button.pack()

# create a button widget to play the output file
play_button = tk.Button(window, text="Play output file", font=("Arial", 14), command=play_output)
play_button.pack()

# disable the play button widget initially
play_button.config(state="disabled")

# create a label widget to display the status of the process
status_label = tk.Label(window, text="", font=("Arial", 14))
status_label.pack()

# define global variables for wav file and spk file names
wav_file = ""
spk_file = ""

# define a global variable for output file name
out_file = ""

# define a function that selects a file using file dialog and updates the global variables and status label
def select_file(file_type):
    """
    Selects a file using file dialog and updates the global variables and status label.

    Parameters:
    file_type (str): the type of the file to select ("wav" or "spk")

    Returns:
    None
    """
    global wav_file
    global spk_file, status_label
    # use file dialog to select a file
    # specify the initial directory according to the file type
    if file_type == "wav":
        initial_dir = WORKING_DIR
    elif file_type == "spk":
        # use os.path to construct the path to the data_svc/singer directory
        initial_dir = os.path.join(WORKING_DIR, "data_svc", "singer")
    else:
        initial_dir = WORKING_DIR
    file = filedialog.askopenfilename(initialdir=initial_dir, title=f"Select {file_type} file")
    # check if the file is valid
    if file:
        # update the global variables and status label according to the file type
        if file_type == "wav":
            wav_file = convert_to_wav(file)
            status_label.config(text=f"Wav file selected: {wav_file}")
        elif file_type == "spk":
            spk_file = file
            status_label.config(text=f"Spk file selected: {spk_file}")
        else:
            status_label.config(text=f"Invalid file type: {file_type}")
    else:
        # no file selected, update the status label accordingly
        status_label.config(text=f"No {file_type} file selected")

# start the main loop of the window
window.mainloop()
