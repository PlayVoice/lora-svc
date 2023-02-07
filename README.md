# vits-singing-voice-conversion
vits singing voice conversion based on ppg &amp; hubert

VI-SVC model is just VITS without MAS and DurationPredictor. Big data [more and more wave] make things to be interesing!


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


# framework

![base_train](/assets/SVC1.png)

![base_infer](/assets/SVC2.png)

![pro_train](/assets/SVC1_pro.png)

![pro_infer](/assets/SVC2_pro.png)

![unix_infer](/assets/SVC2_unix.png)


# train
[VI-SVC](/svc/README.md)

# how to clone your voice
use base model and your voice data to fine tune, just voice data（speech or song） without lables.

# TODO
NSF-VI-SVC based on openai/whisper

https://github.com/PlayVoice/whisper_ppg

