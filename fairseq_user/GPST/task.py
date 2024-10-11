# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from omegaconf import MISSING, II

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task, FairseqTask
from fairseq.data import Dictionary

from .generator import HSequenceGenerator
from .dataset_lmdb import SemanticAcousticDataset

logger = logging.getLogger(__name__)

@dataclass
class SemanticAcousticConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    max_source_positions: int = 650  #13s
    max_target_positions: int = 975
    codec_depth: int = 8
    codec_dim: int = 1024
    target_vocab_size: int = 8192
    source_vocab_size: int = 10000
    reduce: bool = False
    prompt_mode: int = 0  #0 for no prompt, 1 for complete semantic and no acoustic, 2 for half semantic half acoutic, 3 for half semantic but no acoustic
    num_shards: int = -1
    shard_id: int = -1
    semantic_prefix: str = 'xlsr2_unit'
    codec_prefix: str = 'meta24khz_6kpbs_codec'
    
    seed: int = II("common.seed")
    add_bos_token: bool = False
    tokens_per_sample: int = max_source_positions + max_target_positions
        
    
@register_task("semantic_acoustic", dataclass=SemanticAcousticConfig)
class SemanticAcousticTask(FairseqTask):
    cfg: SemanticAcousticConfig
    
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def load_dataset(self, split, epoch = 1, task_cfg = None, **kwargs):
        
        if self.cfg.num_shards > 0:
            assert not split.startswith("train")
        
        if "prompt" in split:
            self.datasets[split] = SemanticAcousticPromptDatasetCreator(
            )
        else:
            self.datasets[split] = SemanticAcousticDataset(
                self.cfg.data,
                split=split,
                semantic_prefix=self.cfg.semantic_prefix,
                codec_prefix=self.cfg.codec_prefix,
                src_dict=self.src_dict,
                tgt_dict=self.tgt_dict,
                codec_dim=self.cfg.codec_dim,
                codec_depth=self.cfg.codec_depth,
                reduce=self.cfg.reduce,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
            )
            
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        src_dict = Dictionary()
        for i in range(cfg.source_vocab_size):
            src_dict.add_symbol(str(i))
        logger.info(f"src dictionary size: " f"{len(src_dict):,}")

        if cfg.target_vocab_size != cfg.codec_depth * cfg.codec_dim:
            logger.info(f"target_vocab_size={cfg.target_vocab_size} != codec_depth={cfg.codec_depth} * codec_dim={cfg.codec_dim}")
            logger.info("infer vocab size")
            cfg.target_vocab_size = cfg.codec_depth * cfg.codec_dim
            
        tgt_dict = Dictionary()
        for i in range(cfg.target_vocab_size):
            tgt_dict.add_symbol(str(i))
        logger.info(f"tgt dictionary size: " f"{len(tgt_dict):,}")
        
        return cls(cfg, src_dict, tgt_dict)

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        seq_gen_cls = HSequenceGenerator
        generator = super().build_generator(
                models,
                args,
                seq_gen_cls,
                extra_gen_cls_kwargs,
            )
        generator.prompt_mode = self.args.prompt_mode
        generator.codec_depth = self.args.codec_depth
        generator.max_target_positions = self.args.max_target_positions
        generator.max_source_positions = self.args.max_source_positions
        return generator
    
    
