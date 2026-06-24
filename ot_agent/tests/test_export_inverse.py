# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The HF export inverts the loader's weight transform; a wrong inverse silently
corrupts the checkpoint (transposed / mis-reshaped attention weights load
"successfully" but the model is broken). These tests round-trip every transform
in the real Qwen3 key map: torch -> (loader forward) -> nnx -> (our inverse) ->
torch must be the identity.
"""

import numpy as np

from tunix.models.qwen3 import model as qm
from tunix.models.qwen3 import params as qp
from tunix.utils.torch_utils import torch_key_to_jax_key

from ot_agent.export_hf import _invert_transform


def _forward(torch_arr, transform):
  """Mirror tunix.models.safetensors_loader: transpose(permute) then reshape."""
  v = torch_arr
  if transform is not None:
    permute, reshape = transform
    if permute:
      v = v.transpose(permute)
    if reshape:
      v = v.reshape(reshape)
  return v


def _tiny_config() -> qm.ModelConfig:
  # Small but structurally faithful Qwen3 (GQA: num_kv_heads < num_heads).
  return qm.ModelConfig(
      num_layers=2, vocab_size=32, embed_dim=16, hidden_dim=24,
      num_heads=4, head_dim=8, num_kv_heads=2, rope_theta=1_000_000, norm_eps=1e-6,
  )


def _torch_shapes(cfg):
  e, nh, hd, nkv, hdn, ff, v = (
      cfg.embed_dim, cfg.num_heads, cfg.head_dim, cfg.num_kv_heads,
      cfg.head_dim, cfg.hidden_dim, cfg.vocab_size,
  )
  return {
      "model.layers.0.self_attn.q_proj.weight": (nh * hd, e),
      "model.layers.0.self_attn.k_proj.weight": (nkv * hdn, e),
      "model.layers.0.self_attn.v_proj.weight": (nkv * hdn, e),
      "model.layers.0.self_attn.o_proj.weight": (e, nh * hd),
      "model.layers.0.mlp.gate_proj.weight": (ff, e),
      "model.layers.0.mlp.up_proj.weight": (ff, e),
      "model.layers.0.mlp.down_proj.weight": (e, ff),
      "model.embed_tokens.weight": (v, e),
      "lm_head.weight": (v, e),
      "model.norm.weight": (e,),
      "model.layers.0.input_layernorm.weight": (e,),
  }


def test_every_transform_roundtrips():
  cfg = _tiny_config()
  key_map = qp._get_key_and_transform_mapping(cfg)
  rng = np.random.default_rng(0)
  for tk, shape in _torch_shapes(cfg).items():
    _, transform = torch_key_to_jax_key(key_map, tk)
    torch_arr = rng.standard_normal(shape).astype(np.float32)
    nnx_arr = _forward(torch_arr, transform)          # what the loader stores
    recovered = _invert_transform(nnx_arr, transform, shape)  # our export inverse
    assert recovered.shape == shape, f"{tk}: shape {recovered.shape} != {shape}"
    np.testing.assert_array_equal(recovered, torch_arr, err_msg=f"{tk} round-trip")


def test_attention_reshape_is_not_a_noop():
  # Guard against an inverse that "passes" only because it never reshapes:
  # q_proj genuinely changes rank (2D torch -> 3D nnx) and back.
  cfg = _tiny_config()
  key_map = qp._get_key_and_transform_mapping(cfg)
  _, transform = torch_key_to_jax_key(key_map, "model.layers.0.self_attn.q_proj.weight")
  shape = (cfg.num_heads * cfg.head_dim, cfg.embed_dim)
  torch_arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
  nnx_arr = _forward(torch_arr, transform)
  assert nnx_arr.ndim == 3  # (embed, heads, head_dim)
  recovered = _invert_transform(nnx_arr, transform, shape)
  np.testing.assert_array_equal(recovered, torch_arr)
