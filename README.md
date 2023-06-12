# singing voice conversion based on whisper & maxgan
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17PK5Vd-oyoxpsZ8nENktFcPZEwZmPrTb?usp=sharing)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/lora-svc">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/lora-svc">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/lora-svc">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/lora-svc">

```
Black technology based on the three giants of artificial intelligence:

OpenAI's whisper, 680,000 hours in multiple languages

Nvidia's bigvgan, anti-aliasing for speech generation

Microsoft's adapter, high-efficiency for fine-tuning
```

https://user-images.githubusercontent.com/16432329/231021007-6e34cbb4-e256-491d-8ab6-5ce4e822da21.mp4


## Dataset preparation

Necessary pre-processing:
- 1 accompaniment separation
- 2 band extension
- 3 sound quality improvement
- 4 cut audio, less than 30 seconds for whisper💗

then put the dataset into the dataset_raw directory according to the following file structure
```shell
dataset_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## Install dependencies

- 1 software dependency
  
  > apt update && sudo apt install ffmpeg
  
  > pip install -r requirements.txt

- 2 download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`

- 3 download whisper model [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), Make sure to download `medium.pt`，put it into `whisper_pretrain/`

- 4 whisper is built-in, do not install it additionally, it will conflict and report an error

## Data preprocessing
- 1， set working directory:

    > export PYTHONPATH=$PWD

- 2， re-sampling

    generate audio with a sampling rate of 16000Hz：./data_svc/waves-16k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000

    generate audio with a sampling rate of 32000Hz：./data_svc/waves-32k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000

- 3， use 16K audio to extract pitch：f0_ceil=900, it needs to be modified according to the highest pitch of your data
    > python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch

    or use next for low quality audio

    > python prepare/preprocess_f0_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch

- 4， use 16K audio to extract ppg
    > python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper

- 5， use 16k audio to extract timbre code
    > python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker

- 6， extract the average value of the timbre code for inference; it can also replace a single audio timbre in generating the training index, and use it as the unified timbre of the speaker for training
    > python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer

- 7， use 32k audio to generate training index
    > python prepare/preprocess_train.py

- 8， training file debugging
    > python prepare/preprocess_zzz.py

```shell
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── whisper
│    └── speaker0
│    │      ├── 000001.ppg.npy
│    │      └── 000xxx.ppg.npy
│    └── speaker1
│           ├── 000001.ppg.npy
│           └── 000xxx.ppg.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

## Train
- 0， if fine-tuning based on the pre-trained model, you need to download the pre-trained model: maxgan_pretrain_32K.pth

    > set pretrain: "./maxgan_pretrain_32K.pth" in configs/maxgan.yaml，and adjust the learning rate appropriately, eg 1e-5

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， start training

    > python svc_trainer.py -c configs/maxgan.yaml -n svc

- 3， resume training

    > python svc_trainer.py -c configs/maxgan.yaml -n svc -p chkpt/svc/***.pth

- 4， view log

    > tensorboard --logdir logs/

## Inference

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， export inference model

    > python svc_export.py --config configs/maxgan.yaml --checkpoint_path chkpt/svc/***.pt

- 3， use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    generate test.ppg.npy; if no ppg file is specified in the next step, generate it automatically

- 4， extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser

    > python pitch/inference.py -w test.wav -p test.csv

- 5，specify parameters and infer

    > python svc_inference.py --config configs/maxgan.yaml --model maxgan_g.pth --spk ./configs/singers/singer0001.npy --wave test.wav --ppg test.ppg.npy --pit test.csv

    when --ppg is specified, when the same audio is reasoned multiple times, it can avoid repeated extraction of audio content codes; if it is not specified, it will be automatically extracted;

    when --pit is specified, the manually tuned F0 parameter can be loaded; if not specified, it will be automatically extracted;

    generate files in the current directory:svc_out.wav

    | args |--config | --model | --spk | --wave | --ppg | --pit | --shift |
    | ---  | --- | --- | --- | --- | --- | --- | --- |
    | name | config path | model path | speaker | wave input | wave ppg | wave pitch | pitch shift |

## Source of code and References
[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

## Contributor

<a href="https://github.com/PlayVoice/lora-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/lora-svc" />
</a>

## Encouragement
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
