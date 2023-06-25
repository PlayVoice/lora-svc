# (place the python script in your lora root directory)
#
# NOTE: for this to work please make sure your directory is setup like this:
#
# trained_models/modelname/
# │
# └── lora_speaker.npy (copy this from the data_svc/ folder)
# │
# └── lora_pitch_statics.npy (copy this from the data_svc/ folder)
# │
# └── maxgan.yaml (copy this from the config/ folder)
# │
# └── maxgan_g.pth (you get this file in the root directory when you run python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/modelname/modelname_00001000.pt)
#
# You can also run the script and type -c in the file name and it will set it up for you
#
# Please avoid using any input files/model names with spaces / special characters as that will probably cause issues
# Make sure what you call the model name matches the name you trained the model as in the chkpt/ folder when you set the name when you train and run this command: python svc_trainer.py -c config/maxgan.yaml -n MODEL_NAME

from pydub import AudioSegment
import shutil
import os

def convert_to_wav(input_file):
    # Create the output filename by replacing the extension
    output_file = os.path.splitext(input_file)[0] + '.wav'

    # Check if the WAV file already exists
    if os.path.exists(output_file):
        print(f"WAV file '{output_file}' already exists. Skipping conversion.")
        return
        
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Set the desired parameters for the output WAV file
    sample_width = 2  # 16-bit
    sample_rate = 48000  # 48kHz

    # Convert the audio to the desired format
    audio = audio.set_frame_rate(sample_rate).set_sample_width(sample_width).set_channels(1)

    # Create the output filename by replacing the extension
    output_file = os.path.splitext(input_file)[0] + '.wav'

    # Export the audio to the WAV file
    audio.export(output_file, format='wav')
    
    # Delete the original file
    #os.remove(input_file)

    print(f"Conversion complete. The WAV file '{output_file}' has been created.")
    
def remove_extension(file_name):
    # Remove file extension from the file name
    base_name = os.path.splitext(file_name)[0]
    return base_name

def replace_filename_in_commands(file_name, first_time, include_statics=False):
    commands = [
        "python svc_inference_ppg.py -w WAV_INPUT.wav -p WAV_INPUT.ppg.npy",
        f"python svc_inference.py --config trained_models/{model_name}/maxgan.yaml --model trained_models/{model_name}/maxgan_g.pth --spk trained_models/{model_name}/lora_speaker.npy --ppg WAV_INPUT.ppg.npy --wave WAV_INPUT.wav"
    ]

    if first_time:
        current_dir = os.getcwd()
        os.environ['PYTHONPATH'] = current_dir

    for command in commands:
        command = command.replace('WAV_INPUT', file_name)

        if include_statics:
            command = command.replace(f"--spk trained_models/{model_name}/lora_speaker.npy", f"--spk trained_models/{model_name}/lora_speaker.npy --statics trained_models/{model_name}/lora_pitch_statics.npy")

        # Get the output file name
        if command.endswith('.npy'):
            output_file = file_name + '.ppg.npy'
        else:
            output_file = None

        # Check if the output file already exists for whisper/inference.py and pitch/inference.py commands
        if output_file and os.path.exists(output_file) and ('inference_ppg' in command):
            print("Skipping command: '{}' as file '{}' already exists.".format(command, output_file))
        else:
            os.system(command)

    # Check if svc_out.wav exists and rename it
    svc_out_file = 'svc_out.wav'
    new_file_name = 'svc_{}_out_{}.wav'.format(model_name, file_name)
    if os.path.exists(svc_out_file):

        # Add _depitched suffix to the renamed output file if -n flag is used
        if include_statics:
            depitched_file_name = new_file_name.replace('.wav', '_depitched.wav')
            os.rename(svc_out_file, depitched_file_name)
            print("Renamed {} to {}".format(new_file_name, depitched_file_name))
        else:
            os.rename(svc_out_file, new_file_name)
            print("Renamed {} to {}".format(svc_out_file, new_file_name))
    
def process_file(file_name_input, first_time):
    if " -n" in file_name_input:
        file_name = file_name_input.replace(" -n", "")
        if "." in file_name:
            file_extension = os.path.splitext(file_name)[-1].lower()
            if file_extension in supported_extensions:
                convert_to_wav(file_name)
                file_name = remove_extension(file_name)
                replace_filename_in_commands(file_name, True, include_statics=True)
            elif file_extension in [".wav"]:
                file_name = remove_extension(file_name)
                replace_filename_in_commands(file_name, True, include_statics=True)
            else:
                file_name = remove_extension(file_name)
                print("Unsupported File Type")
                file_name_input = input("Enter the file name: ")
                process_file(file_name_input, False)
        else:
            # Check if any supported file variations exist in the folder
            found_file = False

            for extension in supported_extensions:
                file_path = file_name + extension
                if os.path.exists(file_path):
                    convert_to_wav(file_path)
                    file_name = remove_extension(file_path)
                    found_file = True
                    replace_filename_in_commands(file_name, True, include_statics=True)
                    break

            if not found_file:
                # Check if WAV file exists
                wav_file_path = file_name + ".wav"
                if os.path.exists(wav_file_path):
                    # Handle WAV file
                    file_name = remove_extension(file_name)
                    replace_filename_in_commands(file_name, True, include_statics=True)
                else:
                    print("Unsupported File Type")
                    file_name_input = input("Enter the file name: ")
                    process_file(file_name_input, False)
    else:
        file_name = file_name_input
        if "." in file_name:
            file_extension = os.path.splitext(file_name)[-1].lower()
            if file_extension in supported_extensions:
                convert_to_wav(file_name)
                file_name = remove_extension(file_name)
                replace_filename_in_commands(file_name, True, include_statics=False)
            elif file_extension in [".wav"]:
                file_name = remove_extension(file_name)
                replace_filename_in_commands(file_name, True, include_statics=False)
            else:
                file_name = remove_extension(file_name)
                print("Unsupported File Type")
                file_name_input = input("Enter the file name: ")
                process_file(file_name_input, False)
        else:
            # Check if any supported file variations exist in the folder
            found_file = False
            
            for extension in supported_extensions:
                file_path = file_name + extension
                if os.path.exists(file_path):
                    convert_to_wav(file_path)
                    file_name = remove_extension(file_path)
                    found_file = True
                    replace_filename_in_commands(file_name, True, include_statics=False)
                    break

            if not found_file:
                # Check if WAV file exists
                wav_file_path = file_name + ".wav"
                if os.path.exists(wav_file_path):
                    # Handle WAV file
                    file_name = remove_extension(file_name)
                    replace_filename_in_commands(file_name, True, include_statics=False)
                else:
                    print("Unsupported File Type")
                    file_name_input = input("Enter the file name: ")
                    process_file(file_name_input, False)

def cleanup_files():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the paths to the folders
    junk_files_dir = os.path.join(script_dir, "junkfiles")
    results_dir = os.path.join(script_dir, "results")
    raw_dir = os.path.join(script_dir, "raw")

    # Create the folders if they don't exist
    os.makedirs(junk_files_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Get a list of all files in the current directory
    files = os.listdir(script_dir)

    # Iterate over the files
    for file in files:
        # Check if the file is a "ppg.npy" file
        if file.endswith("ppg.npy"):
            file_path = os.path.join(script_dir, file)
            # Move the file to the "junkfiles" folder
            shutil.move(file_path, os.path.join(junk_files_dir, file))

        # Check if the file is a WAV file containing "svc_" in its name
        elif file.endswith(".wav") and "svc_" in file:
            file_path = os.path.join(script_dir, file)
            # Move the file to the "results" folder
            shutil.move(file_path, os.path.join(results_dir, file))

    # Get a list of remaining files in the current directory
    files = os.listdir(script_dir)

    # Iterate over the remaining files
    for file in files:
        # Check if the file is a WAV, MP3, M4A, or FLAC file
        if file.endswith((".wav",) + tuple(supported_extensions)):
            file_path = os.path.join(script_dir, file)
            # Move the file to the "raw" folder
            shutil.move(file_path, os.path.join(raw_dir, file))


MODEL_NAME_FILE = "model_name.txt"

supported_extensions = [".mp3", ".m4a", ".ogg", ".aac", ".mp4", ".mov", ".flac"]

def save_model_name(model_name):
    with open(MODEL_NAME_FILE, 'w') as file:
        file.write(model_name)

def load_model_name():
    if os.path.exists(MODEL_NAME_FILE):
        with open(MODEL_NAME_FILE, 'r') as file:
            return file.read().strip()
    return ""

# Load the model name from the file if it exists
model_name = load_model_name()
# Process the initial file
file_name_input = input("Enter the file name (-c for setup/config): ")

if file_name_input == "-c":
    valid_input = False
    while not valid_input:
        setuporconfig = input("Are you creating a setup or switching to a different model? (setup/switch/cleanup): ")
        
        if setuporconfig == "setup":
            model_name = input("Enter your model name: ")
            checkpoint_number = input("Enter checkpoint #: ")
            num_zeros = 8 - len(checkpoint_number)
            checkpoint_filename = f"{model_name}_{'0' * num_zeros}{checkpoint_number}.pt"
            
            # Create trained_models/{model_name} directory if it doesn't exist
            trained_models_dir = "trained_models"
            model_dir = os.path.join(trained_models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            command = f"python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/{model_name}/{checkpoint_filename}"
            os.system(command)
            
            # Move maxgan_g.pth and maxgan_lora.pth files to trained_models/{model_name} folder
            shutil.move("maxgan_g.pth", os.path.join(model_dir, "maxgan_g.pth"))
            shutil.move("maxgan_lora.pth", os.path.join(model_dir, "maxgan_lora.pth"))
            
            # Copy config/maxgan.yaml to trained_models/{model_name} folder
            shutil.copy("config/maxgan.yaml", os.path.join(model_dir, "maxgan.yaml"))
            
            # Copy data_svc/lora_speaker.npy to trained_models/{model_name} folder
            shutil.copy("data_svc/lora_speaker.npy", os.path.join(model_dir, "lora_speaker.npy"))
            
            # Copy data_svc/lora_pitch_statics.npy to trained_models/{model_name} folder
            shutil.copy("data_svc/lora_pitch_statics.npy", os.path.join(model_dir, "lora_pitch_statics.npy"))
            
            # Run the command python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/{model_name}/{checkpoint_filename}
            valid_input = True
            
        elif setuporconfig == "switch":
            model_name = input("Enter your model name: ")
            save_model_name(model_name)
            process_file(file_name_input, True)
            valid_input = True

        elif setuporconfig == "cleanup":
            cleanup_files()
            print("Cleanup completed.")

            valid_input = True
        else:
            print("Invalid Input. Please enter 'setup' or 'switch'.")
            
        file_name_input = input("Enter the file name (without extension): ")
        process_file(file_name_input, True)
    
else:
    process_file(file_name_input, True)

# Process more files
while True:
    answer = input("Do you want to process another file? (y/n): ")
    if answer.lower() == 'n':
        break
    elif answer.lower() == 'y':
        file_name_input = input("Enter the file name (without extension): ")

        process_file(file_name_input, False)

    else:
        print("Invalid input. Please enter 'y' or 'n'.")
