import torch.distributed as distr
import torch
import pathlib
import numpy as np
import lmdb
import shutil
import os
import logging
import tqdm
from pathlib import Path
import torchaudio
import jsonlines
import subprocess

from data_handler import ManifestDataset
from distributed import init_distributed_context
from encodec_reader import EncodecFeatureReader
from seamless_reader import Wav2vecFeatureReader

logger = logging.getLogger(__name__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", 
        type=Path,
        required=True, 
        help="Path to the dataset manifest file"
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument("--distributed_port", type=int, default=58554)

    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args



def transcribe_lmdb(args, rank, world_size):
    dataset = ManifestDataset(args.manifest)


    speech_encoder_enocodec = EncodecFeatureReader(
        bandwidth = args.bandwidth, 
    )
    os.makedirs(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec", exist_ok=True)
    lmdb_path = worker_shard_path(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem, "", rank).as_posix()
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    output_files_encodec = lmdb.open(lmdb_path, map_size=int(1e12))

    speech_encoder_seamless = Wav2vecFeatureReader(
        checkpoint_path = "xlsr2_1b_v2", 
        kmeans_path = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        layer=35,
        dtype=torch.float16 if args.fp16 else torch.float32
    )
    os.makedirs(args.manifest.parent / "xlsr2_unit", exist_ok=True)
    lmdb_path = worker_shard_path(args.manifest.parent / "xlsr2_unit" / args.manifest.stem, "", rank).as_posix()
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    output_files_seamless = lmdb.open(lmdb_path, map_size=int(1e12))
    
    
    for i in tqdm.tqdm(range(rank, len(dataset), world_size)):
        audio_path = dataset[i]
    
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        encoded_encodec = speech_encoder_enocodec(waveform, sr)
        encoded_seamless = speech_encoder_seamless(waveform, sr)
        
        with output_files_encodec.begin(write=True) as txn:
            stream = encoded_encodec['codec'].numpy()
            txn.put(audio_path.as_posix().encode('utf-8'), stream.tobytes())
            
        with output_files_seamless.begin(write=True) as txn:
            stream = encoded_seamless['units'].numpy()
            txn.put(audio_path.as_posix().encode('utf-8'), stream.tobytes())
        
    output_files_encodec.close()
    output_files_seamless.close()
    
    return args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem, args.manifest.parent / "xlsr2_unit" / args.manifest.stem
    
    
def merge_lmdb(full_output, suffix, n_workers):
    env = lmdb.open(full_output.as_posix(), map_size=int(1e12))
    with env.begin(write=True) as txn:
        for worker_id in range(n_workers):
            partial_path = worker_shard_path(full_output, suffix, worker_id)
            lmdb_env = lmdb.open(partial_path.as_posix(), readonly=True)
            with lmdb_env.begin() as lmdb_txn:
                for key, value in lmdb_txn.cursor():
                    txn.put(key, value)
            lmdb_env.close()
    env.close()    
    
    for worker_id in range(n_workers):
        partial_path = worker_shard_path(full_output, suffix, worker_id)
        shutil.rmtree(partial_path)
        print(f"Deleted: {partial_path}")


def worker_shard_path(fname, suffix, worker_id) -> pathlib.Path:
    return fname.with_suffix(f".{suffix}_partial_{worker_id}")


def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        output_file_codec, output_file_seamless = transcribe_lmdb(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()
        
    if context.is_leader:
        merge_lmdb(output_file_codec, "", context.world_size)
        merge_lmdb(output_file_seamless, "", context.world_size)

if __name__ == "__main__":
    args = get_args()
    main(args)
