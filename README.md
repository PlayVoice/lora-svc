# singing voice conversion based on whisper & maxgan, and target to LoRA

```diff
- maxgan v1 == bigvgan + nsf        PlayVoice/lora-svc
- maxgan v2 == bigvgan + latent f0  PlayVoice/maxgan-svc
```


Uni-SVC for **multi-singer** (release state v0.4): branch https://github.com/PlayVoice/lora-svc/tree/uni-svc-multi-singer, experiment on 56 singers

Uni-SVC for **baker** (release state v0.3): branch https://github.com/PlayVoice/lora-svc/tree/uni-svc-baker, experiment on pure speech

Uni-SVC for **Opencpop** (release state v0.2): branch https://github.com/PlayVoice/lora-svc/tree/uni-svc-opencpop


## Train

- 1 download [Multi-Singer](https://github.com/Multi-Singer/Multi-Singer.github.io) data, and change sample rate of waves to 16000Hz, and put waves to **./data_svc/waves**
    > you can do

- 2 download speaker encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), and put **best_model.pth** and **condif.json** into **speaker_pretrain/**

    > python svc_preprocess_speaker.py ./data_svc/waves ./data_svc/speaker

- 3 download whisper [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), and put **medium.pt** into **whisper_pretrain/**

    > python svc_preprocess_ppg.py -w ./data_svc/waves -p ./data_svc/whisper

- 4 extract pitch and generate **filelist/train.txt** & filelist/eval.txt

    > python svc_preprocess_f0.py

- 5 start train

    > python svc_trainer.py -c config/default_c32.yaml -n uni_svc

data tree like this

    data_svc/
    |
    └── ids
    |     └──spk1_encoding.npy
    |     └──spk2_encoding.npy
    |     └──spk3_encoding.npy
    └── pitch
    │     ├── spk1
    │     │   ├── 000001.pit.npy
    │     │   ├── 000002.pit.npy
    │     │   └── 000003.pit.npy
    │     └── spk2
    │         ├── 000001_pit.npy
    │         ├── 000002_pit.npy
    │         └── 000003_pit.npy
    └── speakers
    │     ├── spk1
    │     │   ├── 000001.spk.npy
    │     │   ├── 000002.spk.npy
    │     │   └── 000003.spk.npy
    │     └── spk2
    │         ├── 000001.spk.npy
    │         ├── 000002.spk.npy
    │         └── 000003.spk.npy 
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

## Reference
[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

https://github.com/chenwj1989/pafx

## Data-sets

KiSing      http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/

PopCS 		  https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md

opencpop 	  https://wenet.org.cn/opencpop/download/

Multi-Singer 	https://github.com/Multi-Singer/Multi-Singer.github.io

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

## Awesome opensource singing voice conversion

https://github.com/innnky/so-vits-svc

https://github.com/prophesier/diff-svc

https://github.com/yxlllc/DDSP-SVC

https://github.com/lesterphillip/SVCC23_FastSVC

# Notice
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
