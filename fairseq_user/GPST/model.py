# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import copy
import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from functools import partial
import types

from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    FairseqIncrementalDecoder,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
    Linear,
)
from fairseq.utils import new_arange
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
from fairseq_user.GPST.util import extract_features_with_embedding

logger = logging.getLogger(__name__)

@dataclass
class HLMConfig(TransformerLanguageModelConfig):
    decoder_normalize_before: bool = True
    global_decoder_embed_dim: int = 512
    global_decoder_ffn_embed_dim: int = 2048
    global_decoder_layers: int = 12
    global_decoder_attention_heads: int = 8
    local_decoder_embed_dim: int = 256
    local_decoder_ffn_embed_dim: int = 1024
    local_decoder_layers: int = 4
    local_decoder_attention_heads: int = 4
    
    semantic_loss: bool = True
    share_lm_head: bool = True
    local_drop: float = 0
    
class HTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, global_decoder, local_decoder, projector, dictionary=None):
        super().__init__(dictionary)
        self.global_decoder = global_decoder
        self.local_decoder = local_decoder
        self.global_local_projector = projector
        self.global_decoder.forward = types.MethodType(extract_features_with_embedding, self.global_decoder)
        self.local_decoder.forward = types.MethodType(extract_features_with_embedding, self.local_decoder)
        
    def global_embed_tokens(self, src_tokens, incremental_state=None):
        positions = self.global_decoder.embed_positions(src_tokens[...,0])
        if incremental_state is not None:
            src_tokens = src_tokens[:, -1:]  # only choose the last token for computation, kv cache
            positions = positions[:, -1:]
                
        x = self.global_decoder.embed_scale * self.global_decoder.embed_tokens(src_tokens).sum(-2)
        x = x + positions
        
        return x
        
    def local_embed_tokens(self, prev_output_tokens, first_token, incremental_state=None):
        positions = self.local_decoder.embed_positions(prev_output_tokens, incremental_state) # B, nq, h
        acoustic_token = prev_output_tokens[...,:-1] # B, nq-1
        
        if incremental_state is not None:
            acoustic_token = acoustic_token[:, -1:]
            positions = positions[:, -1:]
                    
        x = self.local_decoder.embed_scale * self.local_decoder.embed_tokens(acoustic_token) # no cumsum beacuase of std
        
        return torch.cat([first_token, x], dim = 1) + positions
        
    def forward(
        self, 
        src_tokens,
        prev_output_tokens_local,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        local_drop=0,
    ):
        global_embeds = self.global_embed_tokens(src_tokens, incremental_state) # B,L,h
        global_hidden_states, global_extra = self.global_decoder(
            global_embeds, 
            src_tokens[...,0], 
            incremental_state=incremental_state,
        ) # B,L,h
        
        semantic_parts = global_hidden_states[(src_tokens[...,0] < len(self.global_decoder.dictionary)) & (src_tokens[...,0].ne(self.global_decoder.padding_idx))]
        acoustic_parts = global_hidden_states[src_tokens[...,0] >= len(self.global_decoder.dictionary)]
        
        if self.global_decoder.output_projection is not None:
            semantic_logits = self.global_decoder.output_projection(semantic_parts) # B,D x D,V
        else:
            semantic_logits = None
        
        mask = prev_output_tokens_local.ne(self.local_decoder.padding_idx).any(-1) # B, L
        prev_output_tokens_local = prev_output_tokens_local[mask] # B * L, nq
        
        if local_drop > 0:
            target_score = mask.clone().float().uniform_() # B, L
            target_score.masked_fill_(~mask, 2.0)
            target_length = mask.sum(1).float()
            target_length = target_length * (1 - local_drop) #target_length.clone().uniform_()
            target_length = target_length + 1  # make sure at least one token undrop.
            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            mask = mask.masked_fill(
                ~target_cutoff.scatter(1, target_rank, target_cutoff), False
            )[mask] # b*l
            prev_output_tokens_local = prev_output_tokens_local[mask] # b*l, nq
            acoustic_parts = acoustic_parts[mask]
                
        local_embeds = self.local_embed_tokens(prev_output_tokens_local, self.global_local_projector(acoustic_parts).unsqueeze(1)) # incremental_state is for global
                        
        local_hidden_states, local_extra = self.local_decoder(
            local_embeds,
            prev_output_tokens_local,
        ) # B,L,9,h
        acoustic_logits = self.local_decoder.output_projection(local_hidden_states)
        
        return semantic_logits, acoustic_logits, prev_output_tokens_local

    def prepare_incremental_state(
        self,
        src_tokens,
        incremental_state,
    ): # compute semantic lv cache at once 
        s_embeds = self.global_decoder.embed_scale * self.encoder_embed_tokens(src_tokens)
        positions = self.global_decoder.embed_positions(src_tokens)
        x = s_embeds + positions
        
        # cannot directly use extract_features_with_embedding because incremental_state will affect self_attn_mask
        alignment_layer = self.global_decoder.num_layers - 1
        if self.global_decoder.layernorm_embedding is not None:
            x = self.global_decoder.layernorm_embedding(x)
        x = self.global_decoder.dropout_module(x)
        x = x.transpose(0, 1)
        self_attn_padding_mask = None
        if self.global_decoder.cross_self_attention or src_tokens.eq(self.global_decoder.padding_idx).any():
            self_attn_padding_mask = src_tokens.eq(self.global_decoder.padding_idx)
        for idx, layer in enumerate(self.global_decoder.layers):
            self_attn_mask = self.global_decoder.buffered_future_mask(x)
            x, layer_attn, _ = layer(
                x,
                None,
                None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
    
    def generate_semantic(
        self,
        src_tokens,
        prev_output_tokens=None,
        encoder_out=None, 
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):        
        positions = self.global_decoder.embed_positions(src_tokens, incremental_state)[:, -1:]
        s_embeds = self.global_decoder.embed_scale * self.encoder_embed_tokens(src_tokens[:,-1:])
        g_embeds = s_embeds + positions
            
        global_hidden_states, extra = self.global_decoder(
            g_embeds, 
            src_tokens[:, -1:], 
            incremental_state=incremental_state,
        ) # B,L,h
        if self.global_decoder.output_projection is not None:
            semantic_logits = global_hidden_states[:, -1:].matmul(self.global_decoder.output_projection) # B,1,D x D,V
        else:
            semantic_logits = None
        return semantic_logits, extra
        
    def generate_acoustic(
        self, 
        src_tokens,
        prev_output_tokens,
        encoder_out=None, 
        is_first=True,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        topk=1, temperature=1
    ):
        if is_first:
            global_embeds = self.global_embed_tokens(src_tokens, prev_output_tokens, incremental_state) # B,L,h
            global_hidden_states, extra = self.global_decoder(
                global_embeds, 
                prev_output_tokens[:, -1:, 0], 
                incremental_state=incremental_state,
            ) # B,L,h
            global_hidden_states = global_hidden_states[:, -1:] # choose the last patch
            
            incremental_state_local = {} # kv cache for local 
            prev_output_tokens_local = torch.zeros(prev_output_tokens.shape[0], 1, device=prev_output_tokens.device).long()  
            local_embeds = self.global_local_projector(global_hidden_states) + self.local_decoder.embed_positions(prev_output_tokens_local, incremental_state_local)
            local_hidden_states, _ = self.local_decoder( # only compute the global hidden states
                local_embeds,
                prev_output_tokens_local,
                incremental_state=incremental_state_local,
            )
            acoustic_logits_local = self.local_decoder.output_projection(local_hidden_states) # B,D x D,V
            extra["incremental_state_local"] = incremental_state_local
            return acoustic_logits_local, extra
        else:
            for i in range(1, prev_output_tokens.shape[-1]):
                local_embeds, positions = self.local_embed_tokens(prev_output_tokens[:,:i + 1], incremental_state) # i+1 for pos emb and i+1 will be cut for token emb
                # only the last token emb will be outputed
                local_embeds += positions
                local_hidden_states, _ = self.local_decoder(
                    local_embeds,
                    prev_output_tokens[:,[i - 1]], # for pad compute
                    incremental_state=incremental_state,
                )
                # topk sampling
                lprobs = (self.local_decoder.output_projection(local_hidden_states[:,-1]) / temperature).log_softmax(-1) # B, nq+4
                lprobs[:, self.local_decoder.dictionary.pad()] = -math.inf  # never select pad
                lprobs[:, self.local_decoder.dictionary.unk()] = -math.inf  # never select unk
                lprobs[:, self.local_decoder.dictionary.eos()] = -math.inf  # never select eos in that eos would not in
                lprobs, top_indices = lprobs.topk(topk)
                indices_buf = torch.multinomial(lprobs.exp_(), 1, replacement=True).view(-1)     
                prev_output_tokens[:,i] = top_indices[torch.arange(lprobs.shape[0], device=lprobs.device), indices_buf] + i * (self.local_decoder.output_projection.out_features - self.local_decoder.dictionary.nspecial)
            return prev_output_tokens      
    
    def max_positions(self):
        return self.global_decoder.max_positions()  
    
    
class StackLMHead(nn.Module):
    def __init__(self, in_dim, out_dim, depth) -> None:
        super().__init__()
        self.weight = nn.Parameter(in_dim ** -0.5 * torch.randn(depth, in_dim, out_dim))
        
    def forward(self, x):
        return x.unsqueeze(-2).matmul(self.weight).squeeze(-2) # B, nq, 1, D x nq, D, V = B,nq,V
    
@register_model("st2at", dataclass=HLMConfig)
class HTransformerLanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder, codec_depth, codec_dim):
        super().__init__(decoder)
        self.args = args
        self.codec_depth = codec_depth
        self.codec_dim = codec_dim
    
    def build_global_decoder(cls, args, task):
        global_decoder_embed_tokens = Embedding(len(task.target_dictionary) + len(task.source_dictionary), args.global_decoder_embed_dim, task.target_dictionary.pad())
        global_config = copy.deepcopy(args)
        global_config.decoder_embed_dim = args.global_decoder_embed_dim
        global_config.decoder_ffn_embed_dim = args.global_decoder_ffn_embed_dim
        global_config.decoder_layers = args.global_decoder_layers
        global_config.decoder_attention_heads = args.global_decoder_attention_heads
        global_config.decoder_output_dim = args.global_decoder_embed_dim
        global_config.decoder_input_dim  = args.global_decoder_embed_dim
        global_config.max_target_positions = task.cfg.max_source_positions + task.cfg.max_target_positions + 2
        
        global_decoder = TransformerDecoder(
            global_config,
            task.source_dictionary,
            global_decoder_embed_tokens,
            no_encoder_attn=True,
        )
        if not args.semantic_loss:
            global_decoder.output_projection = None
        
        try:
            from flash_attn import flash_attn_func
            for i in range(global_decoder.num_layers):
                global_decoder.layers[i].self_attn.forward = types.MethodType(flash_forward, global_decoder.layers[i].self_attn)
        except:
            logger.warning("flash attention not available")
            
        return global_decoder
    
    def build_local_decoder(cls, args, task):
        local_decoder_embed_tokens = Embedding(len(task.target_dictionary), args.local_decoder_embed_dim, task.target_dictionary.pad())
        local_config = copy.deepcopy(args)
        local_config.decoder_embed_dim = args.local_decoder_embed_dim
        local_config.decoder_ffn_embed_dim = args.local_decoder_ffn_embed_dim
        local_config.decoder_layers = args.local_decoder_layers
        local_config.decoder_attention_heads = args.local_decoder_attention_heads
        local_config.decoder_output_dim = args.local_decoder_embed_dim
        local_config.decoder_input_dim  = args.local_decoder_embed_dim
        local_config.max_target_positions = task.cfg.codec_depth
        
        if args.share_lm_head:
            output_projection_acoustic = Linear(local_config.decoder_embed_dim, task.cfg.codec_dim + task.target_dictionary.nspecial, bias=False)
            nn.init.normal_(output_projection_acoustic.weight, 0, local_config.decoder_embed_dim ** -0.5)
        else:
            output_projection_acoustic = StackLMHead(local_config.decoder_embed_dim, task.cfg.codec_dim + task.target_dictionary.nspecial, task.cfg.codec_depth)
        
        local_decoder = TransformerDecoder(
            local_config,
            task.target_dictionary,
            local_decoder_embed_tokens,
            no_encoder_attn=True,
            output_projection=output_projection_acoustic,
        )
        
        try:
            from flash_attn import flash_attn_func
            for i in range(local_decoder.num_layers):
                local_decoder.layers[i].self_attn.forward = types.MethodType(flash_forward, local_decoder.layers[i].self_attn)
        except:
            logger.warning("flash attention not available")
        
        return local_decoder
                
    @classmethod
    def build_model(cls, args, task):    
        global_decoder = cls.build_global_decoder(cls, args, task)
        local_decoder = cls.build_local_decoder(cls, args, task)
        projector = Linear(args.global_decoder_embed_dim, args.local_decoder_embed_dim)
        
        decoder = HTransformerDecoder(global_decoder, local_decoder, projector)

        return cls(args, decoder, task.cfg.codec_depth, task.cfg.codec_dim)
    
    def forward(
        self,
        sample,
        **kwargs
    ):
        src_tokens = sample["net_input"]["src_tokens"]
        src_labels = sample["src_labels"]
        prev_output_tokens_local = sample["net_input"]['prev_output_tokens']
                                  
        semantic_logits, acoustic_logits, target = self.decoder(
            src_tokens,
            prev_output_tokens_local,
            local_drop=self.args.local_drop,
        )
        
        extra = {}
        
        target = (target - torch.arange(self.codec_depth, device=prev_output_tokens_local.device) * self.codec_dim).clip(min=self.decoder.local_decoder.dictionary.eos()) 
        # the final token is eos, which will be negative so clip it to eos
        
        output = {
            "word_ins_acoustic": {
                "out": acoustic_logits.flatten(0,1),
                "nll_loss": True,
                "factor": 1.0,
                "tgt": target.flatten(0,1),
            },
        }
        
        if self.args.semantic_loss:
            output["word_ins_semantic"] = {
                "out": semantic_logits,
                "nll_loss": True,
                "factor": 1.0,
                "tgt": src_labels[src_labels.ne(self.decoder.global_decoder.padding_idx)]
            }
        
        return output, extra
    