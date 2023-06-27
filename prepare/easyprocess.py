# import the subprocess module
import subprocess

# define a list of commands to run
commands = [
    "export PYTHONPATH=$PWD",
    "python3 prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000",
    "python3 prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000",
    "python3 prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch",
    "python3 prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper",
    "python3 prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker",
    "python3 prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer",
    "python3 prepare/preprocess_train.py",
    "python3 prepare/preprocess_zzz.py -c configs/maxgan.yaml"
]

# loop through the commands and run them one by one
for command in commands:
    # call the subprocess.Popen function with the command and shell=True
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # get the output and error streams
    output, error = process.communicate()
    # print the output and error if any
    print(output.decode())
    print(error.decode())
