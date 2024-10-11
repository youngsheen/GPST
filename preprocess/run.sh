SPLIT=test-clean
ROOT=/data3/yongxinzhu

#wget https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
#wget https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/xlsr2_1b_v2.pt
#wget https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy


export TORCH_HOME="/data/yongxinzhu/.cache/torch"
export FAIRSEQ2_CACHE_DIR="/data/yongxinzhu/.cache/fairseq2"

#python preprocess/get_manifest.py \
#    --root $ROOT/LibriSpeech/$SPLIT \
#    --dest $ROOT/LibriSpeech \
#    --ext flac \
#    --name $SPLIT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=6669 \
    preprocess/transcribe.py \
    --manifest $ROOT/LibriSpeech/$SPLIT.tsv \
    --bandwidth 6 \
    --batchsize 16
