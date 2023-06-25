# import tkinter module and other modules
import tkinter as tk
from tkinter import filedialog
import os
import subprocess

# define global variables for working directory, LD_LIBRARY_PATH, model file and checkpoint file 
WORKING_DIR = os.getcwd()
LD_LIBRARY_PATH = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
MODEL_FILE = "maxgan_g.pth"
CHECKPOINT_FILE = ""

# define a function that converts a file to wav format using ffmpeg
def convert_to_wav(file):
    # get the file name and extension
    file_name = os.path.basename(file)
    file_ext = os.path.splitext(file_name)[1]

    # check if the file extension is wav
    if file_ext == ".wav":
        # no need to convert, just return the original file name
        return file_name
    else:
        # use ffmpeg to convert the file to wav format
        new_file_name = os.path.splitext(file_name)[0] + ".wav"
        new_file_path = os.path.join(WORKING_DIR, new_file_name)
        subprocess.run(["ffmpeg", "-i", file, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", new_file_path])
        return new_file_path

# define a function that finds the highest numbered checkpoint file in a given directory
def find_highest_checkpoint(dir_name):
    # initialize the highest number and file name variables
    highest_number = 0
    highest_file = ""

    # loop through all the files in the directory
    for file in os.listdir(dir_name):
        # get the file name and extension
        file_name = os.path.basename(file)
        file_ext = os.path.splitext(file_name)[1]

        # check if the file extension is pt
        if file_ext == ".pt":
            # extract the number from the file name
            number = int(os.path.splitext(file_name)[0].split("_")[-1])

            # compare the number with the highest number so far
            if number > highest_number:
                # update the highest number and file name variables
                highest_number = number
                highest_file = os.path.join(dir_name, file)

    # return the highest file name
    return highest_file

# define a function that takes a wav file and a spk file as arguments and runs all the commands
def run_all(wav_file, spk_file):
    # set working directory and LD_LIBRARY_PATH variables using global variables defined earlier
    os.environ["PYTHONPATH"] = WORKING_DIR
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    # use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage
    subprocess.run(["python3", "whisper/inference.py", "-w", wav_file, "-p", wav_file + ".ppg.npy"])

    # extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser
    subprocess.run(["python3", "pitch/inference.py", "-w", wav_file, "-p", wav_file + ".csv"])

    # specify parameters and infer using the maxgan_g.pth model and checkpoint file names defined earlier as global variables 
    subprocess.run(["python3", "svc_inference.py", "--config", "configs/maxgan.yaml", "--model", MODEL_FILE, "--spk", spk_file, "--wave", wav_file, "--ppg", wav_file + ".ppg.npy", "--pit", wav_file + ".csv"])

    # rename the output file to include the output file name given as an argument 
    output_file = "svc_aaq_out_" + os.path.basename(wav_file) + "_" + os.path.basename(spk_file) + ".wav"
    output_path = os.path.join(WORKING_DIR, output_file)
    os.rename(os.path.join(WORKING_DIR, "svc_out.wav"), output_path)

# create a tkinter window object 
window = tk.Tk()

# set window title and size 
window.title("SVC GUI")
window.geometry("600x400")

# create a label widget to display instructions 
label = tk.Label(window, text="Welcome to SVC GUI! Please follow these steps:\n\n1. Select an input audio file (any format).\n2. Select one or more speaker files (.spk.npy format).\n3. Click on Run All button to start the inference process.\n4. Wait for the process to finish and check the output files (.wav format) in the working directory.", justify=tk.LEFT)
label.pack(padx=10, pady=10)

# create a string variable to store the input file name 
input_file = tk.StringVar()

# create a function that opens a file dialog and sets the input file name 
def select_input_file():
    file = filedialog.askopenfilename(initialdir=WORKING_DIR, title="Select input file")
    input_file.set(file)

# create a button widget to select the input file 
button_input = tk.Button(window, text="Select input file", command=select_input_file)
button_input.pack(padx=10, pady=10)

# create a list variable to store the spk files 
spk_files = []

# create a function that opens a file dialog and appends the spk files to the list 
def select_spk_files():
    files = filedialog.askopenfilenames(initialdir=os.path.join(WORKING_DIR, "configs/singers"), title="Select spk files", filetypes=(("spk files", "*.spk.npy"),))
    spk_files.extend(files)

# create a button widget to select the spk files 
button_spk = tk.Button(window, text="Select spk files", command=select_spk_files)
button_spk.pack(padx=10, pady=10)

# create a function that runs all commands for each spk file in parallel using threads
def run_all_parallel():
    # import threading module
    import threading

    # check if the input file and spk files are not empty
    if input_file.get() and spk_files:
        # convert the input file to wav format and get the new file name
        wav_file = convert_to_wav(input_file.get())

        # export inference model using the checkpoint file name 
        global CHECKPOINT_FILE
        CHECKPOINT_FILE = find_highest_checkpoint(os.path.join(WORKING_DIR, "chkpt/svc"))
        subprocess.run(["python3", "svc_export.py", "--config", "configs/maxgan.yaml", "--checkpoint_path", CHECKPOINT_FILE])

        # loop through each spk file in the list 
        for spk_file in spk_files:
            # create a thread object that runs the run_all function for wav file and spk file 
            thread = threading.Thread(target=run_all, args=(wav_file, spk_file))

            # start the thread 
            thread.start()

        # wait for all threads to finish using join method 
        for thread in threading.enumerate():
            if thread is not threading.main_thread():
                thread.join()

        # show a message box that indicates the process is done 
        tk.messagebox.showinfo("Done", "The inference process is done. Please check the output files in the working directory.")

    else:
        # show a message box that indicates the input file or spk files are missing 
        tk.messagebox.showerror("Error", "Please select an input file and one or more spk files.")

# create a button widget to run all commands in parallel 
button_run = tk.Button(window, text="Run All", command=run_all_parallel)
button_run.pack(padx=10, pady=10)

# start the main loop of the window 
window.mainloop()
