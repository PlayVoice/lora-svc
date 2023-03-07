# UNI-SVC, singing voice conversion based on whisper and UnivNet and NSF

You will feel the beauty of the code from this project.


### Reference
[UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet

https://github.com/openai/whisper/

UNI-SVC will be for [@yoyo鹿鸣_Lumi](https://space.bilibili.com/488836173)

https://user-images.githubusercontent.com/16432329/219535886-8f80346a-143d-47fc-8c84-29e6f5be5143.mp4


DID image to video :https://studio.d-id.com/

### Train
download whisper model: "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"

download opencpop data: https://wenet.org.cn/opencpop/download/

change sample rate of waves, and put waves to ./data_opencpop/waves

> python svc_preprocess_ppg.py -w ./data_opencpop/waves -p ./data_opencpop/whisper

> python svc_preprocess_f0.py

> python svc_trainer.py -c config/default_c32.yaml -n uni_svc

google cloud: [log.zip](https://drive.google.com/file/d/1DKFWs3QBr8Pi4_XcdtENH4m1l2ZoLU2o/view?usp=share_link)

> tensorboard --logdir logs/uni_svc/

<img width="798" alt="loss1000" src="https://user-images.githubusercontent.com/16432329/222940116-777b980f-f2b2-453b-91db-d79cd5441d1a.png">


### Infer
export clean model

> python svc_export.py --config config/default_c32.yaml --checkpoint_path chkpt/uni_svc/uni_svc_0740.pt

download preview form release page

> python svc_inference.py --config config/default_c32.yaml --model uni_svc.pth --wave uni_svc_test.wav

### Preview， It takes longer to train to get good quality.

video from [@一直在吃的周梓琦](https://space.bilibili.com/20473341)

https://www.bilibili.com/video/BV1Kg4y1E77u

change to opencpop, add audio effect

https://user-images.githubusercontent.com/16432329/222939881-ce73e7de-0899-4b96-a459-cf375b6288c0.mp4

video from [@真栗](https://space.bilibili.com/210752)

https:///www.bilibili.com/video/BV1UT411J7Vf


https://user-images.githubusercontent.com/16432329/223148035-7ddc2278-1887-437c-bc27-03a523de1869.mp4


Male to female

https://user-images.githubusercontent.com/16432329/223142419-bdaf7842-2a28-401c-b7c4-5126cee0d931.mp4


### Data-sets
KiSing      http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/

PopCS 		  https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md

opencpop 	  https://wenet.org.cn/opencpop/download/

OpenSinger 	https://github.com/Multi-Singer/Multi-Singer.github.io

M4Singer	  https://github.com/M4Singer/M4Singer/blob/master/apply_form.md


CSD 		    https://zenodo.org/record/4785016#.YxqrTbaOMU4

KSS		      https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset

JVS MuSic	  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music

PJS		      https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus

JUST Song	  https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song


MUSDB18		  https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems

DSD100 		  https://sigsep.github.io/datasets/dsd100.html


Aishell-3 	http://www.aishelltech.com/aishell_3

VCTK 		    https://datashare.ed.ac.uk/handle/10283/2651

# Notice
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。

# VI-SVC
VI-SVC was dropped to https://github.com/PlayVoice/VI-SVC/tree/vi_svc
