# singing voice conversion based on whisper & maxgan, and target to LoRA

```per
maxgan v1 == bigvgan + nsf        PlayVoice/lora-svc

maxgan v2 == bigvgan + latent f0  PlayVoice/maxgan-svc
```
```
基于人工智能三大巨头的黑科技：

来至OpenAI的whisper，68万小时多种语言

来至Nvidia的bigvgan，语音生成抗锯齿

来至Microsoft的adapter，高效率微调
```

下面是基于预训练模型定制专有音色

## 训练

- 1 数据准备，将音频切分小于30S（推荐10S左右/可以不依照句子结尾）， 转换采样率为16000Hz, 将音频数据放到 **./data_svc/waves**
    > 这个我想你会~~~

- 2 下载音色编码器: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 解压文件，把 **best_model.pth** 和 **condif.json** 放到目录 **speaker_pretrain/**

    提取每个音频文件的音色
    
    > python svc_preprocess_speaker.py ./data_svc/waves ./data_svc/speaker
    
    取所有音频音色的平均作为目标发音人的音色
    
    > python svc_preprocess_speaker_lora.py ./data_svc/

- 3 下载whisper模型 [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 确定下载的是**medium.pt**，把它放到文件夹 **whisper_pretrain/** 中，提取每个音频的内容编码

    > python svc_preprocess_ppg.py -w ./data_svc/waves -p ./data_svc/whisper

- 4 提取基音，同时生成训练文件 **filelist/train.txt**，剪切train的前5条用于制作**filelist/eval.txt**

    > python svc_preprocess_f0.py

- 5 从release页面下载预训练模型maxgan_pretrain，放到model_pretrain文件夹中，预训练模型中包含了生成器和判别器

    > python svc_trainer.py -c config/maxgan.yaml -n lora -p model_pretrain/maxgan_pretrain.pth


你的文件目录应该长这个样子~~~

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
    └── waves
    │     ├── 000001.wav
    │     ├── 000002.wav
    │     └── 000003.wav
    └── whisper
          ├── 000001.ppg.npy
          ├── 000002.ppg.npy
          └── 000003.ppg.npy

## egs: 使用50句猫雷、训练十分钟的日志如下
https://user-images.githubusercontent.com/16432329/228889388-d7658930-6187-48a8-af37-74096d41c018.mp4

## 推理
导出生成器，判别器只会在训练中用到

> python svc_inference_export.py --config config/maxgan.yaml --checkpoint_path chkpt/lora/lora_0090.pt

导出的模型在当前文件夹maxgan_g.pth，文件大小为31.6M

> python svc_inference.py --config config/maxgan.yaml --model maxgan_g.pth --spk ./data_svc/**lora_speaker.npy** --wave test.wav

生成文件在当前目录svc_out.wav

**PS.** 本项目集成了音效算法，你可以使用混响等常见音效

啥？生成的音色不太像！
```
待补充~~~
1，发音人音域统计
2，推理音区偏移
```

## 更好的音质
为了训练更高的音质，需要使用分支maxgan_v1_pretrain，需要使用大量语料，重新训练预训练模型

**更高的音质=更深的网络层+更多的通道数+更高的采样率**

下面是一组 16K 采样率、160 hop的更大模型的一组参数示例：

```
gen:
  upsample_rates: [5,4,2,2,2]
  upsample_kernel_sizes: [20,16,4,4,4]
  upsample_initial_channel: 512
  resblock_kernel_sizes: [3,7,11]
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
```

分支代码有差异，根据实际需要选择合理的代码分支。

## 音色融合
天生具备~~~，demo稍等~~~

## 流式推理
whisper改造完成，但是nsf会产生断点~~~继续研究

## 最初的梦想，发音人插件化
![maxgan_svc](https://user-images.githubusercontent.com/16432329/229016002-963f1d70-a5f6-474d-98fa-051bc8c21f26.png)

## 代码来源和参考文献
[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

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
