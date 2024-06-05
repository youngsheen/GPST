# GPST PyTorch Implementation

This is a PyTorch implementation of the paper [Generative Pre-Trained Speech Language Model with Efficient Hierarchical Transformer](https://arxiv.org/abs/2406.00976v1).
```
@misc{zhu2024generative,
      title={Generative Pre-trained Speech Language Model with Efficient Hierarchical Transformer}, 
      author={Yongxin Zhu and Dan Su and Liqiang He and Linli Xu and Dong Yu},
      year={2024},
      eprint={2406.00976},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Overview

Demo page: [https://youngsheen.github.io/GPST/demo](https://youngsheen.github.io/GPST/demo)

The overview of GPST as following picture shows.
![The overview of GPST](pics/model.png)



### Installation
1. Download the code
```shell 
git clone https://github.com/youngsheen/GPST.git
cd GPST
```

2. Install `fairseq` and `encodec` via `pip`. Install [seamless_communication](https://github.com/facebookresearch/seamless_communication) and [encodec](https://github.com/facebookresearch/encodec)

3. Install [`flash-attn`](https://github.com/Dao-AILab/flash-attention) for faster attention computation.


## Preparation

### Dataset
Download the [LibriSpeech](https://www.openslr.org/12) or [LibriLight](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md) dataset and place it in your directory at `$PATH_TO_YOUR_WORKSPACE/datasets`. We use xlsr2_1b_v2 from [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) to extract semantic tokens and [Encodec](https://github.com/facebookresearch/encodec) to extract acoustic tokens. You can set the `bandwidth` to 6kbps or 12 kbps to control the quality of speech resynthesis. We suggest using `bandwidth=12` since the former half of its acoustic tokens are the same as 6kbps.

```shell
SPLIT=test-clean

python preprocess/get_manifest.py \
    --root datasets/librispeech/LibriSpeech/$SPLIT \
    --dest datasets/librispeech \
    --ext flac \
    --name $SPLIT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=6666 \
    preprocess/transcribe.py \
    --manifest datasets/librispeech/$SPLIT.tsv \
    --seamless


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=6666 \
    preprocess/transcribe.py \
    --manifest datasets/librispeech/$SPLIT.tsv \
    --codec --bandwidth 6

```

## Training Scripts 

## Inference Scripts 