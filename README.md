# singing voice conversion based on whisper, UnivNet and NSF

You will feel the beauty of the code from this project. 

Uni-SVC main branch is for singing voice clone based on whisper with speaker encoder and speaker adapter.

Uni-SVC main target is to develop [lora](https://github.com/PlayVoice/Uni-SVC/blob/main/model/generator.py#L12-L44) for SVC.

With lora, maybe clone a singer just need 10 stence after 10 minutes train. Each singer is a plug-in of the base model.

![lora](https://user-images.githubusercontent.com/16432329/225337790-392b958a-67ec-4643-b26a-018ee8e4cf56.jpg)

Model **which contains [56 singers](https://github.com/PlayVoice/lora-svc/tree/main/config/singers) of 50 hours singing data** is training~~~~

You can down preview model **uni_svc_opensinger_0415.pth** at release page.

https://user-images.githubusercontent.com/16432329/227782805-8a45e99a-8905-4ec8-ac20-b75a45cfbc97.mp4

Uni-SVC for **baker** (release state v0.3): branch https://github.com/PlayVoice/Uni-SVC/tree/uni-svc-baker, experiment on pure speech

Uni-SVC for **Opencpop** (release state v0.2): branch https://github.com/PlayVoice/Uni-SVC/tree/uni-svc-opencpop

## Awesome opensource singing voice conversion

https://github.com/innnky/so-vits-svc

https://github.com/prophesier/diff-svc

https://github.com/yxlllc/DDSP-SVC

https://github.com/lesterphillip/SVCC23_FastSVC

## Reference
[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/chenwj1989/pafx

## Train
download whisper model: https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt

download speaker encoder: https://github.com/mozilla/TTS/wiki/Released-Models

Speaker-Encoder by @mueller91	LibriTTS + VCTK + VoxCeleb + CommonVoice

https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3

download [OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io) data:

change sample rate of waves, and put waves to ./data_svc/waves

> python svc_preprocess_ppg.py -w ./data_svc/waves -p ./data_svc/whisper

> python svc_preprocess_ids.py -w ./data_svc/waves -p ./data_svc/ids

> python svc_preprocess_f0.py

> python svc_trainer.py -c config/default_c32.yaml -n uni_svc

data tree like this

    data_svc/
    |
    |
    └── ids
    |     └──spk1_encoding.npy
    |     └──spk2_encoding.npy
    |     └──spk3_encoding.npy
    |
    └── pitch
    │     ├── spk1
    │     │   ├── 000001_pit.npy
    │     │   ├── 000002_pit.npy
    │     │   └── 000003_pit.npy
    │     └── spk2
    │         ├── 000001_pit.npy
    │         ├── 000002_pit.npy
    │         └── 000003_pit.npy
    └── waves
    │     ├── spk1
    │     │   ├── 000001.wav
    │     │   ├── 000002.wav
    │     │   └── 000003.wav
    │     └── spk2
    │         ├── 000001.wav
    │         ├── 000002.wav
    │         └── 000003.wav
    └── whisper
          ├── spk1
          │   ├── 000001.ppg.npy
          │   ├── 000002.ppg.npy
          │   └── 000003.ppg.npy
          └── spk2
              ├── 000001.ppg.npy
              ├── 000002.ppg.npy
              └── 000003.ppg.npy

## Infer
export clean model

> python svc_inference_export.py --config config/default_c32.yaml --checkpoint_path chkpt/uni_svc/uni_svc_0340.pt

you can download model for release page, after model release

> python svc_inference.py --config config/default_c32.yaml --model uni_svc_opensinger_0415.pth --spk ./config/singers/singer0001.npy --wave uni_svc_test.wav

## demos
#### uni-svc on baker with pure speech, trained 340 epoch of 10k steps

https://user-images.githubusercontent.com/16432329/224460286-2c9ad916-ec2d-40a5-944e-2d3d830d2c63.mp4

#### Demos for opencpop dataset, get model from release page of v0.2

video from [@一直在吃的周梓琦](https://space.bilibili.com/20473341)

https://www.bilibili.com/video/BV1Kg4y1E77u

https://user-images.githubusercontent.com/16432329/222939881-ce73e7de-0899-4b96-a459-cf375b6288c0.mp4

video from [@真栗](https://space.bilibili.com/210752)

https:///www.bilibili.com/video/BV1UT411J7Vf

https://user-images.githubusercontent.com/16432329/223148035-7ddc2278-1887-437c-bc27-03a523de1869.mp4

Male to female

https://user-images.githubusercontent.com/16432329/223142419-bdaf7842-2a28-401c-b7c4-5126cee0d931.mp4

#### Release state

https://user-images.githubusercontent.com/16432329/223479839-32963e4c-874f-4e2b-b1fb-28161648480e.mp4


## Data-sets

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
