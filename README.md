# Rewrite this project for UNI-SVC 【代码之美】

VI-SVC was dropped to https://github.com/PlayVoice/VI-SVC/tree/vi_svc

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

### Infer
export clean model

> python svc_export.py --config config/default_c32.yaml --checkpoint_path chkpt/uni_svc/uni_svc_0740.pt

download preview form release page

> python svc_inference.py --config config/default_c32.yaml --model uni_svc.pth --wave uni_svc_test.wav

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
如果您参考了本项目，请您在您项目中列出本项目。【武德】

If you refer to this project, please list it in your project.