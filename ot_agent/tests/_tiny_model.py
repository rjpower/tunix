# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU test fixture: a tiny, non-tied Qwen3 written as fake HF safetensors.

Small enough to load + train + export on CPU in seconds, but structurally
faithful (GQA: num_kv_heads < num_heads, separate lm_head). Reused by the export
round-trip test and the SFT-loop smoke so both run against real tunix machinery.
"""

import json
import os

import ml_dtypes
import numpy as np
import safetensors.numpy as safe_np

CFG = dict(num_layers=2, vocab_size=32, embed_dim=16, hidden_dim=24,
           num_heads=4, head_dim=8, num_kv_heads=2)


def torch_keys_and_shapes() -> dict[str, tuple[int, ...]]:
  e, ff, nh, hd, nkv, v = (CFG["embed_dim"], CFG["hidden_dim"], CFG["num_heads"],
                           CFG["head_dim"], CFG["num_kv_heads"], CFG["vocab_size"])
  ks = {"model.embed_tokens.weight": (v, e), "model.norm.weight": (e,),
        "lm_head.weight": (v, e)}
  for layer in range(CFG["num_layers"]):
    p = f"model.layers.{layer}"
    ks.update({
        f"{p}.self_attn.q_proj.weight": (nh * hd, e),
        f"{p}.self_attn.k_proj.weight": (nkv * hd, e),
        f"{p}.self_attn.v_proj.weight": (nkv * hd, e),
        f"{p}.self_attn.o_proj.weight": (e, nh * hd),
        f"{p}.self_attn.q_norm.weight": (hd,),
        f"{p}.self_attn.k_norm.weight": (hd,),
        f"{p}.input_layernorm.weight": (e,),
        f"{p}.post_attention_layernorm.weight": (e,),
        f"{p}.mlp.gate_proj.weight": (ff, e),
        f"{p}.mlp.up_proj.weight": (ff, e),
        f"{p}.mlp.down_proj.weight": (e, ff),
    })
  return ks


def write_fake_base_model(base: str, seed: int = 0) -> dict[str, np.ndarray]:
  """Writes ``model.safetensors`` + ``config.json`` for the tiny model into ``base``."""
  rng = np.random.default_rng(seed)
  state = {k: rng.standard_normal(s).astype(ml_dtypes.bfloat16)
           for k, s in torch_keys_and_shapes().items()}
  safe_np.save_file(state, os.path.join(base, "model.safetensors"), metadata={"format": "pt"})
  cfg = {
      "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
      "num_hidden_layers": CFG["num_layers"], "vocab_size": CFG["vocab_size"],
      "hidden_size": CFG["embed_dim"], "intermediate_size": CFG["hidden_dim"],
      "num_attention_heads": CFG["num_heads"], "head_dim": CFG["head_dim"],
      "num_key_value_heads": CFG["num_kv_heads"], "rope_theta": 1000000,
      "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
  }
  with open(os.path.join(base, "config.json"), "w") as f:
    json.dump(cfg, f)
  return state
