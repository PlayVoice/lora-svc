# prework
1, speaker embedding extract

https://github.com/PlayVoice/VI-Speaker

2, use discrete hubert

python prepare/preprocess_hubert.py

# visvc
1, python prepare/preprocess_wave.py

2, python prepare/preprocess.py

3, python train.py -c configs/singing_base.json -m singing_base

# infer
1, python visvc_infer_hubert.py -s [waves]

2, python visvc_infer.py -s [waves] -p [ppg&hubert] -e [speaker_embedding]

# about
 VI-SVC model is just VITS without MAS and DurationPredictor. Big data [more and more wave] make things to be interesing!
