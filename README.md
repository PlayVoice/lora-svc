# singing voice conversion based on whisper & maxgan, and target to LoRA

```per
maxgan v1 == bigvgan + nsf        PlayVoice/lora-svc

maxgan v2 == bigvgan + latent f0  PlayVoice/maxgan-svc
```
```
基于人工智能三大巨头的黑科技：

来至OpenAI的whispe，68万小时多语言

来至Nvidia的bigvgan，语音生成抗锯齿

来至Microsoft的adapter，高效率微调
```

使用自己的数据从头训练，使用分支：https://github.com/PlayVoice/lora-svc/tree/maxgan_v1_pretrain

主分支用于，说明如何基于预训练模型微调定制专有音色；各分支代码有差异，根据您的需要选择合理的代码分支。


## 训练

- 1 数据准备，将音频切分小于30S（推荐10S左右/可以不依照句子结尾）， 转换采样率为16000Hz, 将音频数据放到 **./data_svc/waves**
    > 这个我想你会~~~

- 2 下载音色编码器: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 解压文件，把 **best_model.pth** 和 **condif.json** 放到目录 **speaker_pretrain/**

    提取每个音频文件的音色
    
    > python svc_preprocess_speaker.py ./data_svc/waves ./data_svc/speaker
    
    取所有音频文件的音色的平均作为目标发音人的音色
    
    > python svc_preprocess_speaker_lora.py ./data_svc/

- 3 下载whisper模型 [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 确定下载的是**medium.pt**，把它放到文件夹**whisper_pretrain/**中，提取每个音频的内容编码

    > python svc_preprocess_ppg.py -w ./data_svc/waves -p ./data_svc/whisper

- 4 提取基音，同时生成训练文件 **filelist/train.txt**，剪切train的前5条用于制作**filelist/eval.txt**

    > python svc_preprocess_f0.py

- 5 从release页面下载预训练模型maxgan_pretrain，放到model_pretrain文件夹中，预训练模型中包含了生成器和判别器

    > python svc_trainer.py -c config/maxgan.yaml -n lora -p model_pretrain/maxgan_pretrain.pth


你的文件目录应该长这个样子~~~

    data_svc/
    |
    └── lora_speaker.npy
    |
    └── pitch
    │     ├── 000001.pit.npy
    │     ├── 000002.pit.npy
    │     └── 000003.pit.npy
    └── speakers
    │     ├── 000001.spk.npy
    │     ├── 000002.spk.npy
    │     └── 000003.spk.npy
    └── waves
    │     ├── 000001.wav
    │     ├── 000002.wav
    │     └── 000003.wav
    └── whisper
          ├── 000001.ppg.npy
          ├── 000002.ppg.npy
          └── 000003.ppg.npy

## 训练实例，使用50句猫雷的效果如下

## 推理
导出生成器，判别器只会在训练中用到

> python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/lora/lora_0090.pt

到出的模型在当前文件夹maxgan_g.pth，文件大小为31.6M

> python svc_inference.py --config config/maxgan.yaml --model maxgan_g.pth --spk ./data_svc/**lora_speaker.npy** --wave test.wav

生成文件在当前目录svc_out.wav

**PS.** 本项目集成了音效算法，你可以使用混响等常见音效

## 代码来源和参考文献
[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

https://github.com/chenwj1989/pafx

# 注意事项
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
