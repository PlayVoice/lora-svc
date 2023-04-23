# singing voice conversion based on whisper & maxgan
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17PK5Vd-oyoxpsZ8nENktFcPZEwZmPrTb?usp=sharing)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/lora-svc">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/lora-svc">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/lora-svc">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/lora-svc">

[【中文说明】](README_zh_CN.md)

```
Black technology based on the three giants of artificial intelligence:

OpenAI's whisper, 680,000 hours in multiple languages

Nvidia's bigvgan, anti-aliasing for speech generation

Microsoft's adapter, high-efficiency for fine-tuning
```
Open source plug-in singing voice library model, based on the LoRA principle:

<p align="center">
  <img src="https://user-images.githubusercontent.com/16432329/225337790-392b958a-67ec-4643-b26a-018ee8e4cf56.jpg" />
</p>

Train the model from scratch based on a large amount of data, using the branch: [lora-svc-for-pretrain](https://github.com/PlayVoice/lora-svc/tree/lora-svc-for-pretrain)

https://user-images.githubusercontent.com/16432329/231021007-6e34cbb4-e256-491d-8ab6-5ce4e822da21.mp4

The following is the customization process base on pre-trained model.

## Train

- 1 Data preparation, segment the audio into less than 30S (recommended about 10S/you need not follow the end of the sentence)

    convert the sampling rate to `16000Hz`, and put the audio data in `./data_svc/waves-16k`

    convert the sampling rate to `48000Hz`, and put the audio data in `./data_svc/waves-48k`

    > I think you can handle~~~

- 2 Download the timbre encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3) , unzip the file, put `best_model.pth.tar` into the directory `speaker_pretrain/`

    Extract the timbre of each audio file
    
    > python svc_preprocess_speaker.py ./data_svc/waves-16k ./data_svc/speaker

- 3 Download the whisper model multiple [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), make sure the download is `medium.pt` , put it in the folder `whisper_pretrain/` , and extract the content code of each audio

    > sudo apt update && sudo apt install ffmpeg

    > python svc_preprocess_ppg.py -w ./data_svc/waves-16k -p ./data_svc/whisper

- 4 Extract the pitch and generate the training file `filelist/train.txt` at the same time, cut the first 5 items of the train to make `filelist/eval.txt`

    > python svc_preprocess_f0.py

- 5 Take the average of all audio timbres as the timbre of the target speaker, and complete the sound field analysis
    
    > python svc_preprocess_speaker_lora.py ./data_svc/

    Generate two files, lora_speaker.npy and lora_pitch_statics.npy

- 6 Download the pre-training model [maxgan_pretrain_48K_5L.pth](https://github.com/PlayVoice/lora-svc/releases/tag/v0.5.5) from the release page and put it in the `model_pretrain` folder. The pre-training model contains the generator and the discriminator

    https://github.com/PlayVoice/lora-svc/blob/622fafca87d877a89717aeb09337afbadd885941/config/maxgan.yaml#L17

    > python svc_trainer.py -c config/maxgan.yaml -n lora
    
    Resume training
    
    > python svc_trainer.py -c config/maxgan.yaml -n lora -p chkpt/lora/***.pth
    
    Check state
    
    > tensorboard --logdir logs/


Your file directory should look like this~~~

    data_svc/
    │
    └── lora_speaker.npy
    │
    └── lora_pitch_statics.npy
    │
    └── pitch
    │     ├── 000001.pit.npy
    │     ├── 000002.pit.npy
    │     └── 000003.pit.npy
    └── speakers
    │     ├── 000001.spk.npy
    │     ├── 000002.spk.npy
    │     └── 000003.spk.npy
    └── waves-16k
    │     ├── 000001.wav
    │     ├── 000002.wav
    │     └── 000003.wav
    └── waves-48k
    │     ├── 000001.wav
    │     ├── 000002.wav
    │     └── 000003.wav
    └── whisper
          ├── 000001.ppg.npy
          ├── 000002.ppg.npy
          └── 000003.ppg.npy

## Train LoRA

https://github.com/PlayVoice/lora-svc/blob/d3a1df57e6019c12513bb34e1bd5c8162d5e5055/config/maxgan.yaml#L16


https://github.com/PlayVoice/lora-svc/blob/d3a1df57e6019c12513bb34e1bd5c8162d5e5055/utils/train.py#L34-L35

How to use

- Use for extremely low resources to prevent overfitting

- Use for plug-in sound library development

- Do not use for other

## egs: log of using 50 sentences and training for ten minutes
https://user-images.githubusercontent.com/16432329/228889388-d7658930-6187-48a8-af37-74096d41c018.mp4

## Inference
- 1 Export the generator, the discriminator will only be used in training

    > python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/lora/lora_0090.pt

    The exported model is in the current folder `maxgan_g.pth`, the file size is 54.3M ; `maxgan_lora.pth` is the fine-tuning module, the file size is 0.94M.

- 2 Use whisper to extract content encoding; One-key reasoning is not used, in order to reduce the occupation of memory.

    > python svc_inference_ppg.py -w test.wav -p test.ppg.npy

    out file is test.ppg.npy；If the ppg file is not specified in the next step, the next step will automatically generate it.

- 3 Specify parameters and inference

    > python svc_inference.py --config config/maxgan.yaml --model maxgan_g.pth --spk ./data_svc/`lora_speaker.npy` --wave test.wav

    The generated file is in the current directory `svc_out.wav`; at the same time, `svc_out_pitch.wav` is generated to visually display the pitch extraction results.

**What** ? The resulting sound is not quite like it!

- 1 Statistics of the speaker's vocal range

    Step 5 of training generates: lora_pitch_statics.npy

- 2 Inferring with the range offset

    Specify the pitch parameter:
    
    > python svc_inference.py --config config/maxgan.yaml --model maxgan_g.pth --spk ./data_svc/lora_speaker.npy --statics ./data_svc/lora_pitch_statics.npy --wave test.wav


## Frequency extension：16K->48K `No Need For 48K Ver.`

> python svc_bandex.py -w svc_out.wav

Generate svc_out_48k.wav in the current directory

## Sound Quality Enhancement

Download the pretrained vocoder-based enhancer from the [DiffSinger Community Vocoder Project](https://openvpi.github.io/vocoders) and extract it to a folder `nsf_hifigan_pretrain/`.

NOTE: You should download the zip file with `nsf_hifigan` in the name, not `nsf_hifigan_finetune`.

Copy the svc_out_48k.wav generated after frequency expansion to path\to\input\wavs, run

> python svc_val_nsf_hifigan.py

Generate enhanced files in path\to\output\wavs

## Source of code and References
[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

https://github.com/brentspell/hifi-gan-bwe

https://github.com/openvpi/DiffSinger

https://github.com/chenwj1989/pafx

## Contributor

<a href="https://github.com/PlayVoice/lora-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/lora-svc" />
</a>

## Encouragement
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
