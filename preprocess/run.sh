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
