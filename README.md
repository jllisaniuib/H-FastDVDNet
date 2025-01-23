
# Hybrid FastDVDnet (H-FastDVDNet)

A hybrid video denoising approach, where a lightweight neural network uses, as additional input, 
a pre-denoised sequence obtained with a traditional model-based denoising algorithm (SPTWO).

Authors: A. Buades, P. Céspedes, J.L. Lisani, M. Sánchez
 
Paper submitted at ICIP2025


## Overview

This repository provides a PyTorch implementation of the H-FastDVDnet video denoising algorithm.


## Code User Guide

### Dependencies

Create a conda environment with all the needed dependencies: 
```
conda env create -f requirements.txt -n condaHFastDVDNet
```

Activate environment:
```
conda activate condaHFastDVDNet
```

### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
python test_hfastdvdnet.py --model_file logs/net.pth --max_num_fr_per_seq 100 --test_path data/noisy_data --classic_denoised_path data/pre-denoised_data --target_path data/original_data --save_path data/denoised_data
```


### Training

If you want to train your own models you can execute

```
python train_hfastdvdnet.py --batch_size 16 --no_orthog --trainset_dir_noisy data/noisy_data --trainset_dir_denoised data/pre-denoised_data --trainset_dir_original data/original_data --valset_dir_noisy data/noisy_data_validation --valset_dir_denoised data/pre-denoised_data_validation --valset_dir_original data/original_data_validation --log_dir logs --save_every_epochs 1 --noise_ival 10 30 --val_noiseL 20 --resume_training
```

**NOTES**
* The frames of the sequences to be denoised should be stored under 'data/noisy_data', one sequence per folder ('sequence1', 'sequence2', etc) 
* The frames of the pre-denoised sequences should be stored under 'data/pre-denoised_data', one sequence per folder ('sequence1', 'sequence2', etc)
* The frames of the clean sequences (without AWGN noise added) should be stored under 'data/original_data', one sequence per folder ('sequence1', 'sequence2', etc) 
* The frames in the validation set must follow the same conventions described above 
* The frames of the denoised sequences will be stored under 'denoised_data', one sequence per folder ('sequence1', 'sequence2', etc) 
* In the paper, the SPTWO method was used for pre-denoising, but any other method could have been used (e.g. BM3D)
* The model has been trained for values of noise in [10, 30]
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence


## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Pau Céspedes, based on code from Matias Tassano (https://github.com/pablodz/fastdvdnet)
* Licence   : GPL v3+, see GPLv3.txt


