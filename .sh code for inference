convert_to_wav() {
  # get the file name and extension
  file_name=$(basename $1)
  file_ext=${file_name##*.}

  # check if the file extension is wav
  if [ $file_ext = "wav" ]; then
    # no need to convert, just return the original file name
    echo $file_name
  else
    # use ffmpeg to convert the file to wav format
    new_file_name=${file_name%.*}.wav
    ffmpeg -i $1 -acodec pcm_s16le -ac 1 -ar 16000 $new_file_name
    echo $new_file_name
  fi
}

# define a function that finds the highest numbered checkpoint file in a given directory
find_highest_checkpoint() {
  # get the directory name
  dir_name=$1

  # initialize the highest number and file name variables
  highest_number=0
  highest_file=""

  # loop through all the files in the directory
  for file in $dir_name/*; do
    # get the file name and extension
    file_name=$(basename $file)
    file_ext=${file_name##*.}

    # check if the file extension is pt
    if [ $file_ext = "pt" ]; then
      # extract the number from the file name
      number=${file_name%.pt}
      number=${number##*_}

      # compare the number with the highest number so far
      if [ $number -gt $highest_number ]; then
        # update the highest number and file name variables
        highest_number=$number
        highest_file=$file
      fi
    fi
  done

  # return the highest file name
  echo $highest_file
}

# define a function that takes a wav file as an argument and runs all the commands
run_all() {
  # set working directory and LD_LIBRARY_PATH variables using global variables defined earlier
  export PYTHONPATH=$WORKING_DIR
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

  # use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage
  python3 whisper/inference.py -w $1 -p $1.ppg.npy

  # extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser
  python3 pitch/inference.py -w $1 -p $1.csv

  # specify parameters and infer using the maxgan_g.pth model and checkpoint file names defined earlier as global variables 
  python3 svc_inference.py --config configs/maxgan.yaml --model $MODEL_FILE --spk ./configs/singers/aaq20minute.spk.npy --wave $1 --ppg $1.ppg.npy --pit $1.csv

  # rename the output file to include the output file name given as an argument 
  mv svc_out.wav $2

  # ask the user if they want to delete the wav file 
  read -p "Do you want to delete the wav file ${1}? (y/n): " answer 
  case $answer in 
    [Yy]* ) rm ${1};;
    [Nn]* ) ;;
    * ) echo "Please answer y or n.";;
  esac 
}

# set global variables for working directory, LD_LIBRARY_PATH, model file and checkpoint file 
WORKING_DIR=$PWD 
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH 
MODEL_FILE=maxgan_g.pth 
CHECKPOINT_FILE=$(find_highest_checkpoint chkpt/svc)

# export inference model using the checkpoint file name 
python3 svc_export.py --config configs/maxgan.yaml --checkpoint_path $CHECKPOINT_FILE

# find all the spk files in the configs/singers directory and store them in an array 
spk_files=($(find configs/singers -name "*.spk.npy"))

# loop through each spk file in the array 
for spk_file in ${spk_files[@]}; do

  # ask the user for the input file name and convert it to wav format if needed 
  read -p "Enter the input file name (with extension) for spk ${spk_file}: " input_file 
  wav_file=$(convert_to_wav $input_file)

  # generate an output file name based on the input file name and spk file name 
  output_file=svc_aaq_out_${input_file%.*}_${spk_file##*/}.wav

  # run all commands for wav file and output file in background using & operator 
  run_all $wav_file $output_file &

done

# wait for all background processes to finish using wait command 
wait

