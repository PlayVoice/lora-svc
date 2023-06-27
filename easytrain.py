# import the subprocess module
import subprocess

# define a list of commands to run
commands = [
    "if fine-tuning based on the pre-trained model, you need to download the pre-trained model: maxgan_pretrain_32K.pth",
    "set pretrain: \"./maxgan_pretrain_32K.pth\" in configs/maxgan.yaml，and adjust the learning rate appropriately, eg 1e-5",
    "export PYTHONPATH=$PWD",
    "python svc_trainer.py -c configs/maxgan.yaml -n svc"
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
