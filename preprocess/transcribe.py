# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.distributed as distr
import torch
import pathlib
import numpy as np
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
        "--codec",
        action="store_true",
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--seamless",
        action="store_true",
    )

    parser.add_argument("--distributed_port", type=int, default=58554)

    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args


def worker_shard_path(fname, suffix, worker_id) -> pathlib.Path:
    return fname.with_suffix(f".{suffix}_partial_{worker_id}")


def transcribe(args, rank, world_size):
    dataset = ManifestDataset(args.manifest)

    if args.codec:
        speech_encoder = EncodecFeatureReader(
            bandwidth = args.bandwidth, 
        )
        os.makedirs(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec", exist_ok=True)
        output_files = jsonlines.open(worker_shard_path(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem, "jsonl", rank), mode="w")
    elif args.seamless:
        speech_encoder = Wav2vecFeatureReader(
            checkpoint_path = "xlsr2_1b_v2", 
            kmeans_path = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
            layer=35
        )
        os.makedirs(args.manifest.parent / "xlsr2_unit", exist_ok=True)
        output_files = jsonlines.open(worker_shard_path(args.manifest.parent / "xlsr2_unit" / args.manifest.stem, "jsonl", rank), mode="w")
    else:
        raise NotImplementedError


    for i in tqdm.tqdm(range(rank, len(dataset), world_size)):
        audio_path = dataset[i]
    
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        encoded = speech_encoder(waveform, sr)
        
        if args.codec:
            stream = encoded['codec'].tolist()
            stream = {"name": audio_path.as_posix(), "codec": stream} 
        
        if args.seamless:
            stream = encoded['units'].tolist()
            stream = {"name": audio_path.as_posix(), "unit": stream} 
            
        output_files.write(stream)
    output_files.close()
    
    if args.codec:
        return args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem
    else:
        return args.manifest.parent / "xlsr2_unit" / args.manifest.stem
    
    
def merge_files(full_output, suffix, n_workers):
    output = full_output.with_suffix(f".{suffix}")
    
    run_list = ["cat"]
    for worker_id in range(n_workers):
        partial_path = worker_shard_path(full_output, suffix, worker_id)
        run_list.append(partial_path.as_posix())
    run_list.append(">")
    run_list.append(output.as_posix())
    subprocess.run(" ".join(run_list), shell=True, check=True)
    for worker_id in range(n_workers):
        partial_path = worker_shard_path(full_output, suffix, worker_id)
        partial_path.unlink()


def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        output_file = transcribe(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()
        
    if context.is_leader:
        merge_files(output_file, "jsonl", context.world_size)

if __name__ == "__main__":
    args = get_args()
    main(args)
