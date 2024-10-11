# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import torch
import random
import lmdb
import pickle
from torch.utils.data import Dataset
from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset

logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.n_sample = txn.stat()['entries'] // 2 # half len
    
    def __len__(self):
        return self.n_sample
        
    def sizes_and_keys(self):
        sizes = []
        keys = []
        with self.env.begin(write=False) as txn:
            for key, value in txn.cursor():
                if key.decode('utf-8').endswith("_size"):
                    sizes.append(pickle.loads(value))
                else:
                    keys.append(key)
        return np.array(sizes), keys
    
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            value = np.frombuffer(txn.get(index), dtype=np.int64)
        return value.copy()
    
    def close(self):
        self.env.close()


def random_crop_match(semantic, acoustic, max_src_len, max_tgt_len):
    
    src_tgt_ratio = max_tgt_len / max_src_len
    
    if len(semantic) <= max_src_len:  # direct return below 13s
        return semantic[:max_src_len], acoustic[:max_tgt_len]
    else:
        start = random.randint(0, len(semantic) - max_src_len) # crop to 13s
        return semantic[start: start + max_src_len], acoustic[int(start * src_tgt_ratio): int(start * src_tgt_ratio) + max_tgt_len]


class SemanticAcousticDataset(FairseqDataset):
    def __init__(
        self,
        root,
        split: str,
        semantic_prefix: str = "xlsr2_unit",
        codec_prefix: str = "meta24khz_12kpbs_codec",
        shuffle: bool = True,
        src_dict: Dictionary = None,
        tgt_dict: Dictionary = None,
        codec_dim = None,
        codec_depth = None,
        reduce = None,
        max_source_positions=None,
        max_target_positions=None,
    ):
        
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.split = split
        self.reduce = reduce
        
        self.shuffle = shuffle
                
        self.codec_dim = codec_dim
        self.codec_depth = codec_depth
        
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        
        
        self.units = LMDBDataset(os.path.join(root, semantic_prefix, split))
        self.codecs = LMDBDataset(os.path.join(root, codec_prefix, split))
        
        logger.info('calculating sizes')
        self.src_lens, self.keys = self.units.sizes_and_keys()
        self.tgt_lens, _ = self.codecs.sizes_and_keys()
        self.acoustic_depth = self.tgt_lens[0,0]
        assert self.acoustic_depth >= self.codec_depth
        self.src_lens = self.src_lens[:,-1].clip(max=max_source_positions)
        self.tgt_lens = self.tgt_lens[:,-1].clip(max=max_target_positions) # nq, L -> L
        
        self.n_samples = len(self.units)
                
        logger.info(
            f'split="{split}", n_samples={self.n_samples:_}, shuffle={self.shuffle}, reduce={self.reduce}' 
        )
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_lens[index]

    def size(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_lens[index]

    @property
    def sizes(self):
        return self.src_lens
    
    def __len__(self):
        return self.n_samples

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))

        return order[np.argsort(self.src_lens[order], kind="mergesort")]

    def __getitem__(self, index: int): 
        
        key_name = self.keys[index]       
    
        source = torch.from_numpy(self.units[key_name]).long()
        target = torch.from_numpy(self.codecs[key_name]).reshape(self.acoustic_depth, -1)[:self.codec_depth].transpose(0,1).long() # L, nq
    
        source, target = random_crop_match(source, target, self.max_source_positions, self.max_target_positions) # 1000 = 20s
                   
        if self.reduce:
            source, dur = torch.unique_consecutive(source, return_counts=True)
            
        source = source + self.src_dict.nspecial
        source = torch.cat([source, source.new(1).fill_(self.src_dict.eos())], dim = 0)
        source = source.unsqueeze(1).repeat(1, self.codec_depth)
        source[:,1:] = self.src_dict.pad()
        # [L,8]
        # x,1,1
        # x,1,1
        # eos,1,1
        
        target = target + torch.arange(target.shape[-1]) * self.codec_dim + self.tgt_dict.nspecial
        target = torch.cat([target, target.new(1, target.shape[-1]).fill_(self.tgt_dict.eos())], dim = 0)
        # [L,8]
        # x,x,x
        # x,x,x
        # eos,eos,eos
        
        return {
            "id": index,
            "source": source,
            "target": target,
            
        }
    
    def _collate_source(self, samples) -> torch.Tensor:
    
        src_tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.cat([x["source"][:-1], x["target"][-1:] + len(self.src_dict), x["target"][:-1] + len(self.src_dict)], dim = 0) for x in samples], 
            batch_first=True, padding_value=self.src_dict.pad())  #b,L,nq
        
        prepend_eos = src_tokens.new(src_tokens.shape[0], 1, src_tokens.shape[2]).fill_(self.src_dict.eos())
        prepend_eos[...,1:] = self.src_dict.pad()
        src_tokens = torch.cat([
                prepend_eos,
                src_tokens
        ], dim = 1)
        
        # eos, 1, 1
        # x,1,1
        # eos, eos, eos, + 10000
        # x,x,x, + 10000
        return src_tokens

    def _collate_target(self, samples) -> torch.Tensor:
        target = torch.nn.utils.rnn.pad_sequence([x["target"] for x in samples], batch_first=True, padding_value=self.tgt_dict.pad())
        src_labels = torch.nn.utils.rnn.pad_sequence([x["source"][:,0] for x in samples], batch_first=True, padding_value=self.src_dict.pad())
   
        target_lengths = torch.tensor(
            [x["target"].size(0) for x in samples], dtype=torch.long
        )
        # x,x,x
        # eos,eos,eos
        return target, src_labels, target_lengths
        
    def collater(
        self, samples, return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        
        src_lengths = torch.tensor([x["source"].size(0) for x in samples], dtype=torch.long)
        src_lengths, order = src_lengths.sort(descending=True)
        
        indices = torch.tensor([x["id"] for x in samples], dtype=torch.long)
        indices = indices.index_select(0, order)
        
        src_tokens = self._collate_source(samples)
        src_tokens = src_tokens.index_select(0, order)
        
        target, src_labels, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        src_labels = src_labels.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)

        ntokens = sum(x["target"].size(0) for x in samples)

        src_texts = [self.src_dict.string(samples[i]["source"]) for i in order]

        net_input = {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": target,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "src_labels": src_labels,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "src_texts": src_texts,
        }
        if return_order:
            out["order"] = order
        return out
   
