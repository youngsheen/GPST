SPLIT=test-other
ROOT=/data3/yongxinzhu

export TORCH_HOME="/data/yongxinzhu/.cache/torch"
export FAIRSEQ2_CACHE_DIR="/data/yongxinzhu/.cache/fairseq2"

python preprocess/get_manifest.py \
    --root $ROOT/LibriSpeech/LibriSpeech/$SPLIT \
    --dest $ROOT/LibriSpeech \
    --ext flac \
    --name $SPLIT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --master_port=6669 \
    preprocess/transcribe1.py \
    --manifest $ROOT/LibriSpeech/$SPLIT.tsv \
    --bandwidth 6 --fp16
