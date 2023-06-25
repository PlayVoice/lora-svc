import os
import shutil

def check_file_exists(path):
    return os.path.isfile(path)

def set_config_pretrain(use_pretrain):
    config_path = "config/maxgan.yaml"
    if use_pretrain:
        pretrain_path = "model_pretrain/maxgan_pretrain_48K_5L.pth"
    else:
        pretrain_path = ""
    with open(config_path, 'r') as f:
        lines = f.readlines()
    with open(config_path, 'w') as f:
        for line in lines:
            if line.startswith("  pretrain:"):
                f.write(f"  pretrain: '{pretrain_path}'\n")
            else:
                f.write(line)

def delete_files_and_folders():
    # Delete existing data folders and files
    folders_to_delete = ["data_svc/pitch", "data_svc/speaker", "data_svc/waves-16k", "data_svc/waves-48k", "data_svc/whisper"]
    files_to_delete = ["data_svc/lora_pitch_statics.npy", "data_svc/lora_speaker.npy"]

    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    for file in files_to_delete:
        if os.path.isfile(file):
            os.remove(file)
            
def resume_training(name):
    
    checkpoint_dir = f"chkpt/{name}"
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory 'chkpt/{name}' does not exist. Please check the name and try again.")
        prompt_choice()

    latest_checkpoint = max(os.listdir(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    command = f"python svc_trainer.py -c config/maxgan.yaml -n {name} -p {checkpoint_path}"
    os.system(command)

def start_training(name):
    # Delete existing data folders and files
    existing_files_and_folders = []
    folders_to_check = ["data_svc/pitch", "data_svc/speaker", "data_svc/waves-16k", "data_svc/waves-48k", "data_svc/whisper"]
    files_to_check = ["data_svc/lora_pitch_statics.npy", "data_svc/lora_speaker.npy"]

    for folder in folders_to_check:
        if os.path.exists(folder):
            existing_files_and_folders.append(folder)
    
    for file in files_to_check:
        if os.path.isfile(file):
            existing_files_and_folders.append(file)
    
    if existing_files_and_folders:
        print("The following files and/or folders already exist:")
        for item in existing_files_and_folders:
            print(item)
        
        choice = input("Do you want to continue and delete the existing files/folders? (y/n): ")
        if choice == 'y':
            delete_files_and_folders()
        else:
            print("Exiting...")
            exit()

    commands = [
        "python svc_preprocess_wav.py --out_dir ./data_svc/waves-16k --sr 16000",
        "python svc_preprocess_wav.py --out_dir ./data_svc/waves-48k --sr 48000",
        "python svc_preprocess_speaker.py ./data_svc/waves-16k ./data_svc/speaker",
        "python svc_preprocess_ppg.py -w ./data_svc/waves-16k -p ./data_svc/whisper",
        "python svc_preprocess_f0.py"
    ]

    # Check if ffmpeg is already installed
    ffmpeg_installed = os.system("ffmpeg -version") == 0

    if not ffmpeg_installed:
        commands.insert(3, "sudo apt update && sudo apt install -y ffmpeg")

    train_file = "./filelists/train.txt"
    eval_file = "./filelists/eval.txt"
    
    for command in commands:
        os.system(command)
    
    commands.clear()
    
    print("\nThe preprocessing could take a while depending on the length of your dataset...\n")

    with open(train_file, 'r') as train_f:
        train_lines = train_f.readlines()

    with open(eval_file, 'w') as eval_f:
        eval_f.writelines(train_lines[:5])

    with open(train_file, 'w') as train_f:
        train_f.writelines(train_lines[5:])  # Remove the first 5 lines

    commands.extend([
        "python svc_preprocess_speaker_lora.py ./data_svc/",
        f"python svc_trainer.py -c config/maxgan.yaml -n {name}"
    ])

    for command in commands:
        os.system(command)

def check_dependencies(pretrain):
    missing_dependency = False
    if pretrain:
        if not check_file_exists("model_pretrain/maxgan_pretrain_48K_5L.pth"):
            print("\nPlease download the pretrained model file (maxgan_pretrain_48K_5L.pth) from the following link and place it in the 'model_pretrain' folder:")
            print("https://github.com/PlayVoice/lora-svc/releases/download/v0.5.6/maxgan_pretrain_48K_5L.pth")
            missing_dependency = True

    if not check_file_exists("speaker_pretrain/best_model.pth.tar"):
        print("\nPlease download the 'best_model.pth.tar' file from the following link and place it in the 'speaker_pretrain' folder:")
        print("https://drive.google.com/uc?export=download&id=1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E")
        missing_dependency = True
   
    if not check_file_exists("whisper_pretrain/medium.pt"):
        print("\nPlease download the 'medium.pt' file from the following link and place it in the 'whisper_pretrain' folder:")
        print("https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        missing_dependency = True
    
    if missing_dependency:
        print("\nPlease install your dependencies then rerun this program.")
        exit()

def pretrain_choice(pchoice):
    if pchoice == "start":
        use_pretrain = input("Do you want to use pretraining? (y/n): ")
    else:
        use_pretrain = input("Did you use pretraining on this model? (y/n): ")
        
    if use_pretrain == 'y':
        set_config_pretrain(True)
        check_dependencies(True)
    elif use_pretrain == 'n':
        set_config_pretrain(False)
        check_dependencies(False)
    else:
        print("Invalid choice.")
        pretrain_choice()

def prompt_choice():
    choice = input("Do you want to start a new training session or resume a training session? (start/resume): ")

    if choice == "resume":
        name = input("Please enter the name of the training session: ")
        pretrain_choice(choice)
        resume_training(name)
    elif choice == "start":
        name = input("Please enter the name for the new training session: ")
        pretrain_choice(choice)
        start_training(name)
    else:
        print("Invalid choice.")
        prompt_choice()

def main():
    current_dir = os.getcwd()
    os.environ['PYTHONPATH'] = current_dir
    
    prompt_choice()

if __name__ == "__main__":
    main()
