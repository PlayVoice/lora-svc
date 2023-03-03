# Rewrite this project for UNI-SVC 【代码之美】

VI-SVC was dropped to https://github.com/PlayVoice/VI-SVC/tree/vi_svc

### Reference
[UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet

UNI-SVC will be for [@yoyo鹿鸣_Lumi](https://space.bilibili.com/488836173)

https://user-images.githubusercontent.com/16432329/219535886-8f80346a-143d-47fc-8c84-29e6f5be5143.mp4

DID image to video :https://studio.d-id.com/

### Train
download opencpop data: https://wenet.org.cn/opencpop/download/

change sample rate of waves, and put waves to ./data_opencpop/waves

> python svc_preprocess_ppg.py -w ./data_opencpop/waves -p ./data_opencpop/whisper

> python svc_preprocess_f0.py

> python svc_trainer.py -c config/default_c32.yaml -n uni_svc

<img width="820" alt="train1" src="https://user-images.githubusercontent.com/16432329/222594390-bc4df450-5aac-4bca-9ac7-ecb5ca9dd53b.png">

<img width="844" alt="train2" src="https://user-images.githubusercontent.com/16432329/222594408-b0cad412-721a-4fc4-a725-01c2c0ade444.png">

https://user-images.githubusercontent.com/16432329/222599560-b36cfe6d-186e-49db-bb96-5325991406c9.mp4

# data-sets
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

