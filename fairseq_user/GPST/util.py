# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import MultiheadAttention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    FLASH_AVAILABLE=False

logger = logging.getLogger(__name__)


def flash_forward(
    self,
    query,
    key: Optional[Tensor],
    value: Optional[Tensor],
    key_padding_mask: Optional[Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    need_weights: bool = True,
    static_kv: bool = False,
    attn_mask: Optional[Tensor] = None,
    before_softmax: bool = False,
    need_head_weights: bool = False,
    full_context_alignment=False,
) -> Tuple[Tensor, Optional[Tensor]]:

    if incremental_state is not None:
        saved_state = self._get_input_buffer(incremental_state)
        if saved_state is not None and "prev_key" in saved_state:
            # previous time steps are cached - no need to recompute
            # key and value if they are static
            if static_kv:
                assert self.encoder_decoder_attention and not self.self_attention
                key = value = None
    else:
        saved_state = None

    tgt_len, bsz, embed_dim = query.size()
    src_len = tgt_len
    if key is not None:
        src_len, key_bsz, _ = key.size()
        if not torch.jit.is_scripting():
            assert value is not None
            assert src_len, key_bsz == value.shape[:2]
    

    if self.self_attention:
        key = query
        value = query
    
    q = self.q_proj(query)
    k = self.k_proj(key)
    v = self.v_proj(value)

    if self.bias_k is not None:
        assert self.bias_v is not None
        k, v, attn_mask, key_padding_mask = self._add_bias(
            k, v, attn_mask, key_padding_mask, bsz
        )
    
    def split_heads(x):
        return (
            x.contiguous()
            .view(-1, bsz, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

    kv_bsz = bsz  # need default value for scripting
    massage = split_heads
    q = massage(q)
    if k is not None:
        k = massage(k)
    if v is not None:
        v = massage(v)
    
    
    if saved_state is not None:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            kv_bsz = _prev_key.size(0)
            prev_key = _prev_key.view(kv_bsz, -1, self.num_heads, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
            src_len = k.size(2)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            assert kv_bsz == _prev_value.size(0)
            prev_value = _prev_value.view(kv_bsz, -1, self.num_heads, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        prev_key_padding_mask: Optional[Tensor] = None
        if "prev_key_padding_mask" in saved_state:
            prev_key_padding_mask = saved_state["prev_key_padding_mask"]
        assert k is not None and v is not None
        key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=prev_key_padding_mask,
            batch_size=kv_bsz,
            src_len=k.size(1),
            static_kv=static_kv,
        )

        saved_state["prev_key"] = k
        saved_state["prev_value"] = v
        saved_state["prev_key_padding_mask"] = key_padding_mask
        # In this branch incremental_state is never None
        assert incremental_state is not None
        incremental_state = self._set_input_buffer(incremental_state, saved_state)

    #y = self.attention(q, k, v, **kwargs)
    
    if full_context_alignment:
        y = flash_attn_func(q, k, v, dropout_p=self.dropout_module.p if self.training else 0., causal=False)
    else:
        y = flash_attn_func(q, k, v, dropout_p=self.dropout_module.p if self.training else 0., causal=False if incremental_state is not None and tgt_len == 1 else True)

    y = (
        y.view(bsz, tgt_len, self.num_heads, self.head_dim)
        .flatten(start_dim=2, end_dim=3)
        .transpose(0, 1)
    )
    assert list(y.size()) == [tgt_len, bsz, embed_dim]

    # Dropout not needed because already applied in attention.
    # It is applied to the attention weights before matmul with v.
    y = self.out_proj(y)

    # TODO: support returning attention weights if needed.
    return y, None


 
    
def extract_features_with_embedding(
    self, 
    x, 
    prev_output_tokens, 
    encoder_out=None,
    incremental_state=None,
    full_context_alignment=False,
    alignment_layer=None,
    alignment_heads=None,
):
    if alignment_layer is None:
        alignment_layer = self.num_layers - 1

    enc: Optional[Tensor] = None
    padding_mask: Optional[Tensor] = None
    
    if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
        enc = encoder_out["encoder_out"][0]
    if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
        padding_mask = encoder_out["encoder_padding_mask"][0]
    
    if self.layernorm_embedding is not None:
        x = self.layernorm_embedding(x)
    x = self.dropout_module(x)

    x = x.transpose(0, 1)
    self_attn_padding_mask: Optional[Tensor] = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    # decoder layers
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]
    for idx, layer in enumerate(self.layers):
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        x, layer_attn, _ = layer(
            x,
            enc,
            padding_mask,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        if layer_attn is not None and idx == alignment_layer:
            attn = layer_attn.float().to(x)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if self.layer_norm is not None:
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        x = self.project_out_dim(x)

    return x, {"attn": [attn], "inner_states": inner_states}


