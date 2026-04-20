# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
try:
    from transformers.deepspeed import HfDeepSpeedConfig
except ImportError:
    # For newer transformers versions where deepspeed was moved/removed
    try:
        from deepspeed.utils import HfDeepSpeedConfig
    except ImportError:
        HfDeepSpeedConfig = None
from transformers import LlamaForCausalLM, LlamaConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        if HfDeepSpeedConfig is not None:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True)

    # llama use eos_token_id but not end_token_id
    # Handle case where eos_token_id might be a list (e.g., Llama-3.2)
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else tokenizer.eos_token_id
    
    model.config.end_token_id = eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model
