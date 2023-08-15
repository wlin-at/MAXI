# MAtch, eXpand and Improve: Unsupervised Finetuning for Zero-Shot Action Recognition with Language Knowledge (ICCV 2023)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://wlin-at.github.io/maxi)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.08914)
[![author](https://img.shields.io/badge/Author-Profile-f39f37?color=f39f37)](https://wlin-at.github.io/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/match-expand-and-improve-unsupervised/zero-shot-action-recognition-on-kinetics)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-kinetics?p=match-expand-and-improve-unsupervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/match-expand-and-improve-unsupervised/zero-shot-action-recognition-on-charades-1)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-charades-1?p=match-expand-and-improve-unsupervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/match-expand-and-improve-unsupervised/zero-shot-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-hmdb51?p=match-expand-and-improve-unsupervised)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/match-expand-and-improve-unsupervised/zero-shot-action-recognition-on-ucf101)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-ucf101?p=match-expand-and-improve-unsupervised)


## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.  
Charades dataset [download](https://prior.allenai.org/projects/charades)  
Moments in Time dataset [download](http://moments.csail.mit.edu/)  
UAV Human dataset [download](https://github.com/sutdcv/UAV-Human)  
MiniSSv2: a subset of SSv2, labels and validation splits are provided in `labels/minissv2_labels.csv` and `datasets_splits/ssv2_splits`. See [DATASETS.md](docs/DATASETS.md) for downloading SSv2.  


# Unsupervised Finetuning on K400
Specify the training data root directory, train/val file paths and the path of CLIP ViT-B-16 model in `configs/zero_shot/train/k400/maxi.yaml`. Further, specficy the training hyparameters in `train.sh`.  
In `train.sh`, we specify the train file `datasets_splits/k400_splits/clip_match_result_thresh0.9.txt`, which is the CLIP matching result on K400 with the text bag filtering ratio of 0.9.   

Run `train.sh` for unsupervised finetuning with multiple instance learning using combined text bag of BLIP verbs (data specified via `caption_bag_dir` in `train.sh`) and GPT3 verbs (data specified via `gpt3_bag_dir` in `train.sh`). Due to limitation of space, we only provide a small subset of the GPT3 verb bag and BLIP verb bag in the directory of `data`. The complete verb bags can be downloaded [here](https://files.icg.tugraz.at/d/3b7204bf164044b3aa27/).  

The model is trained on 4x A6000 GPUs.  

```
bash train.sh
```

# Evaluating models
Specify the test data root directory, train/val file paths and the path of CLIP ViT-B-16 model in the several test config files in `configs/zero-shot/eval`. If a test dataset has several splits, there is an individual test config file for each split. Then, Specify the saved model path in `eval.sh`.  
Run `eval.sh` for zero-shot inference on all the seven action datasets. 

```
bash eval.sh
```
At the end of `eval.sh`, `utils/eval_summary.py` is called to summarize the results across different splits of the seven downstream datasets.   

# Citation
Thanks for citing our paper:
```bibtex
@inproceedings{lin2023match,
  title={Match, expand and improve: Unsupervised finetuning for zero-shot action recognition with language knowledge},
  author={Lin, Wei and Karlinsky, Leonid and Shvetsova, Nina and Possegger, Horst and Kozinski, Mateusz and Panda, Rameswar and Feris, Rogerio and Kuehne, Hilde and Bischof, Horst},
  booktitle={ICCV},
  year={2023},
}
```


# Acknowledgements
Some codes are based on [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) and [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP). 
