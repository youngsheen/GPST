<h3 align="center"><a href="https://arxiv.org/abs/2406.00976">
Generative Pre-Trained Speech Language Model with Efficient Hierarchical Transformer</a></h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2406.00976-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2406.00976)

</h5>


<details open><summary>💡 Some other speech AI projects from our team may interest you ✨. </summary><p>
<!--  may -->

> [**Talk With Human-like Agents: Empathetic Dialogue Through Perceptible Acoustic Reception and Reaction**](https://github.com/Haoqiu-Yan/PerceptiveAgent) <br>
> Haoqiu Yan#, Yongxin Zhu#, Kai Zheng, Bing Liu, Haoyu Cao, Deqiang Jiang, Linli Xu <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Haoqiu-Yan/PerceptiveAgent)  [![github](https://img.shields.io/github/stars/Haoqiu-Yan/PerceptiveAgent.svg?style=social)](https://github.com/Haoqiu-Yan/PerceptiveAgent) [![arXiv](https://img.shields.io/badge/Arxiv-2406.12707-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.12707) <br>

</p></details>

# GPST PyTorch Implementation

This is a PyTorch implementation of the paper [Generative Pre-Trained Speech Language Model with Efficient Hierarchical Transformer](https://aclanthology.org/2024.acl-long.97/#).


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

2. Install `fairseq` and `encodec` via `pip`. Install [seamless_communication](https://github.com/facebookresearch/seamless_communication) and [fairseq2](https://github.com/facebookresearch/fairseq2).


3. [Optional] Install [`flash-attn`](https://github.com/Dao-AILab/flash-attention) for faster attention computation.


## Preparation

### Dataset
Download the [LibriSpeech](https://www.openslr.org/12) or [LibriLight](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md) dataset and place it in your directory at `$PATH_TO_YOUR_WORKSPACE/datasets`. We use xlsr2_1b_v2 from [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) to extract semantic tokens and [Encodec](https://github.com/facebookresearch/encodec) to extract acoustic tokens. You can set the `bandwidth` to 6kbps or 12 kbps to control the quality of speech resynthesis. We suggest using `bandwidth=12` since the former half of its acoustic tokens are the same as 6kbps. The scripts will generate a `manifest` containing the path of all files, two lmdb folders containing semantic tokens and acoustic tokens separately.

```shell
bash preprocess/run.sh
```

## Training Scripts 

```shell

OUTPUT_DIR=outputs
ROOT=PATH
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=4,5 torchrun --nnodes=1 --nproc_per_node=2 --master_port=36666 \
    $(which fairseq-hydra-train) --config-dir config \
    --config-name st2at \
    hydra.run.dir=$ROOT/gpst \
    hydra.output_subdir=$OUTPUT_DIR \
    hydra.job.name=$OUTPUT_DIR/train \
    common.tensorboard_logdir=$OUTPUT_DIR/tb \
    checkpoint.save_dir=$OUTPUT_DIR/checkpoints \
    +task.data=$ROOT/LibriSpeech \
 
```

## Inference Scripts 

### TTS

### Voice Conversion


## Citation

If you find GPST useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{zhu-etal-2024-generative,
    title = "Generative Pre-trained Speech Language Model with Efficient Hierarchical Transformer",
    author = "Zhu, Yongxin  and
      Su, Dan  and
      He, Liqiang  and
      Xu, Linli  and
      Yu, Dong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.97",
    doi = "10.18653/v1/2024.acl-long.97",
    pages = "1764--1775",
}
```