import torch.distributed as distr
import torch
import pathlib
import numpy as np
import lmdb
import shutil
import os
import logging
import torch.utils
import tqdm
from pathlib import Path
import pickle
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_handler import ManifestDataset, RankBatchSampler, custom_collate_fn
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
        "--batchsize",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--numworker",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--encoder_max_sample",
        type=int,
        default=128,
    )
    parser.add_argument("--distributed_port", type=int, default=58554)

    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args

def extract_tokens(speech_encoder, batch, output_file):
    if batch['waveform'] is not None:
        encoded_tokens = speech_encoder(batch['waveform'], batch['sample_rate'])
        with output_file.begin(write=True) as txn:
            for stream, audio_path in zip(encoded_tokens['tokens'], batch['audio_path']):
                txn.put(audio_path.encode('utf-8'), stream.numpy().tobytes())
                txn.put(f"{audio_path}_size".encode('utf-8'), pickle.dumps(tuple(stream.size())))                
    
    if batch['waveform_not_chunk'] is not None:
        encoded_tokens = [speech_encoder(wave, [sr]) for wave, sr in zip(batch['waveform_not_chunk'], batch['sample_rate_not_chunk'])]
        with output_file.begin(write=True) as txn:
            for stream, audio_path in zip(encoded_tokens, batch['audio_paths_not_chunk']):
                txn.put(audio_path.encode('utf-8'), stream['tokens'][0].numpy().tobytes())
                txn.put(f"{audio_path}_size".encode('utf-8'), pickle.dumps(tuple(stream['tokens'][0].size())))       

def transcribe_lmdb(args, rank, world_size):
    dataset = ManifestDataset(args.manifest)
    sampler = RankBatchSampler(dataset, args.batchsize, rank, world_size)
    dataloader = DataLoader(dataset, num_workers=args.numworker, batch_sampler=sampler, collate_fn=custom_collate_fn)

    speech_encoder_enocodec = EncodecFeatureReader(bandwidth = args.bandwidth, encoder_max_sample=args.encoder_max_sample)
    os.makedirs(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec", exist_ok=True)
    lmdb_path = worker_shard_path(args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem, "", rank).as_posix()
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    output_files_encodec = lmdb.open(lmdb_path, map_size=int(1e12))

    speech_encoder_seamless = Wav2vecFeatureReader(
        checkpoint_path = "xlsr2_1b_v2", 
        kmeans_path = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",        
        layer=35,
        encoder_max_sample=args.encoder_max_sample
    )
    os.makedirs(args.manifest.parent / "xlsr2_unit", exist_ok=True)
    lmdb_path = worker_shard_path(args.manifest.parent / "xlsr2_unit" / args.manifest.stem, "", rank).as_posix()
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    output_files_seamless = lmdb.open(lmdb_path, map_size=int(1e12))
    
    executor = ThreadPoolExecutor(max_workers=2)
    for batch in tqdm.tqdm(dataloader, total=(len(sampler) + sampler.batch_size - 1) // sampler.batch_size):        
        futures = []
        future_encodec = executor.submit(extract_tokens, speech_encoder_enocodec, batch, output_files_encodec)
        futures.append(future_encodec)
        future_seamless = executor.submit(extract_tokens, speech_encoder_seamless, batch, output_files_seamless)
        futures.append(future_seamless)
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Job Error: {e}")
    
    output_files_encodec.close()
    output_files_seamless.close()
    executor.shutdown(wait=True)
    
    return args.manifest.parent / f"meta24khz_{args.bandwidth}kpbs_codec" / args.manifest.stem, args.manifest.parent / "xlsr2_unit" / args.manifest.stem
    
    
def merge_lmdb(full_output, suffix, n_workers):
    if os.path.exists(full_output):
        shutil.rmtree(full_output)
    env = lmdb.open(full_output.as_posix(), map_size=int(1e12))
    for worker_id in range(n_workers):
        partial_path = worker_shard_path(full_output, suffix, worker_id)
        lmdb_env = lmdb.open(partial_path.as_posix(), readonly=True)
        with lmdb_env.begin() as lmdb_txn:
            with env.begin(write=True) as txn:
                for key, value in tqdm.tqdm(lmdb_txn.cursor()):
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
