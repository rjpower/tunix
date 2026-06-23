"""Load a stock HuggingFace Qwen3 checkpoint into tunix's native flax.nnx Qwen3.

Adapted from the ``tunix-delphi-rl`` loader, generalised to large Qwen3 releases
(Qwen3-8B) by threading the memory-control knobs the 8B SFT needs:

  * ``param_dtype`` separate from the compute ``dtype`` -- we keep params in
    **fp32** for SFT (an 8e-5 AdamW update is below bf16 ULP for unit-scale
    weights, so bf16 storage silently zeroes most updates) while computing in
    bf16.
  * ``remat`` -- ``RematConfig.DECODER`` rematerializes each decoder layer's
    activations, the single biggest HBM lever for fitting 8B at long sequence.
  * ``use_flash_attention`` -- the TPU splash-attention kernel, needed once the
    sequence length grows past a few k.

Qwen3-8B (and -8B-Base) is a *standard* Qwen3: ``rope_theta=1_000_000`` (== the
tunix ``apply_rope`` default) and **no** ``rope_scaling``, so the stock tunix
RoPE is already exact and we apply NO monkeypatch (unlike Delphi). The HF config
matches tunix's built-in ``ModelConfig.qwen3_8b`` preset exactly.
"""

import glob
import json
import os
import struct

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3 import model as qm
from tunix.models.qwen3 import params as qp
from tunix.utils.torch_utils import torch_key_to_jax_key
from transformers import AutoTokenizer


def qwen3_config_from_hf(
    model_dir: str,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype | None = None,
    remat: qm.RematConfig = qm.RematConfig.NONE,
    use_flash_attention: bool = False,
    flash_attention_block_size: int = 1024,
) -> qm.ModelConfig:
  """Builds the tunix ``ModelConfig`` from a stock Qwen3 ``config.json``.

  Maps HF field names to tunix's (``hidden_size``->``embed_dim``,
  ``intermediate_size``->``hidden_dim``, ``num_hidden_layers``->``num_layers``,
  ``num_key_value_heads``->``num_kv_heads``, ``rms_norm_eps``->``norm_eps``,
  ``tie_word_embeddings``->``use_tied_embedding``) and wires the memory knobs.

  Args:
    model_dir: snapshot dir containing ``config.json``.
    dtype: compute dtype for activations.
    param_dtype: storage dtype for params (defaults to ``dtype``; pass fp32 for SFT).
    remat: activation rematerialization policy (DECODER recommended for 8B).
    use_flash_attention: use the TPU splash-attention kernel.
    flash_attention_block_size: block size for the flash kernel.

  Returns:
    A ``qm.ModelConfig`` matching the checkpoint.

  Raises:
    ValueError: if the config is not a Qwen3 architecture, or declares a
      ``rope_scaling`` (stock tunix qwen3 cannot express it).
  """
  with open(os.path.join(model_dir, "config.json"), "r") as f:
    c = json.load(f)

  arch = c.get("architectures") or []
  if c.get("model_type") != "qwen3" and "Qwen3ForCausalLM" not in arch:
    raise ValueError(
        f"qwen3_config_from_hf expects a Qwen3 model; got model_type="
        f"{c.get('model_type')!r}, architectures={arch!r}."
    )
  if c.get("rope_scaling"):
    raise ValueError(
        f"config declares rope_scaling={c['rope_scaling']!r}; stock tunix qwen3 "
        "cannot express it (Qwen3-8B has none, so this should not fire)."
    )

  head_dim = c.get("head_dim") or (c["hidden_size"] // c["num_attention_heads"])
  return qm.ModelConfig(
      num_layers=c["num_hidden_layers"],
      vocab_size=c["vocab_size"],
      embed_dim=c["hidden_size"],
      hidden_dim=c["intermediate_size"],
      num_heads=c["num_attention_heads"],
      head_dim=head_dim,
      num_kv_heads=c["num_key_value_heads"],
      rope_theta=int(c.get("rope_theta", 1_000_000)),
      norm_eps=float(c.get("rms_norm_eps", 1e-6)),
      use_tied_embedding=bool(c.get("tie_word_embeddings", False)),
      remat_config=remat,
      use_flash_attention=use_flash_attention,
      flash_attention_block_size=flash_attention_block_size,
      dtype=dtype,
      param_dtype=param_dtype or dtype,
  )


def _safetensors_keys(model_dir: str) -> list[str]:
  """Reads tensor names from a (possibly sharded) safetensors checkpoint header."""
  shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
  if not shards:
    raise ValueError(f"No .safetensors files in {model_dir}.")
  keys: list[str] = []
  for path in shards:
    with open(path, "rb") as f:
      header_len = struct.unpack("<Q", f.read(8))[0]
      header = json.loads(f.read(header_len).decode("utf-8"))
    keys.extend(k for k in header if k != "__metadata__")
  return keys


def _assert_key_coverage(model_dir: str, config: qm.ModelConfig) -> list[str]:
  """Asserts every safetensors tensor maps to a model param via the key-map.

  The tunix loader only *logs* a warning on an unmapped key, which would silently
  leave a param at random init. We run the same config-dependent key-map ourselves
  and raise on any miss.
  """
  keys = _safetensors_keys(model_dir)
  key_map = qp._get_key_and_transform_mapping(config)
  unmapped = []
  for k in keys:
    try:
      torch_key_to_jax_key(key_map, k)
    except ValueError:
      unmapped.append(k)
  if unmapped:
    raise ValueError(
        f"{len(unmapped)}/{len(keys)} safetensors keys did not map via the qwen3 "
        f"key-map. Unmapped: {unmapped}"
    )
  return keys


def _assert_all_params_concrete(model: qm.Qwen3) -> None:
  """Asserts no model param still holds an abstract ``eval_shape`` sentinel."""
  _, state = nnx.split(model)
  pure = state.to_pure_dict()
  abstract = []

  def _check(path, leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct) or not isinstance(leaf, jax.Array):
      abstract.append(".".join(str(getattr(p, "key", p)) for p in path))
    return leaf

  jax.tree_util.tree_map_with_path(_check, pure)
  if abstract:
    raise ValueError(
        f"{len(abstract)} params never written from checkpoint: {abstract}"
    )


def load_qwen3(
    model_dir: str,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype | None = None,
    remat: qm.RematConfig = qm.RematConfig.NONE,
    use_flash_attention: bool = False,
) -> qm.Qwen3:
  """Loads a stock Qwen3 checkpoint into a tunix Qwen3 with hard coverage checks.

  Applies NO RoPE monkeypatch: stock tunix ``apply_rope`` (theta 1e6, no scaling)
  is already correct for a standard Qwen3.

  Args:
    model_dir: snapshot dir with ``config.json`` + ``*.safetensors``.
    mesh: optional device mesh for sharding the loaded params.
    dtype: compute dtype.
    param_dtype: storage dtype for params (defaults to ``dtype``; pass fp32 for SFT).
    remat: activation rematerialization policy.
    use_flash_attention: use the TPU splash-attention kernel.

  Returns:
    A live ``qm.Qwen3`` nnx module, fully populated.

  Raises:
    ValueError: on incomplete key coverage or any param left abstract.
  """
  config = qwen3_config_from_hf(
      model_dir,
      dtype=dtype,
      param_dtype=param_dtype or dtype,
      remat=remat,
      use_flash_attention=use_flash_attention,
  )
  _assert_key_coverage(model_dir, config)
  # The safetensors loader materializes params at ``config.param_dtype``.
  model = qp.create_model_from_safe_tensors(
      file_dir=model_dir, config=config, mesh=mesh, dtype=config.param_dtype
  )
  _assert_all_params_concrete(model)
  return model


def load_qwen3_tokenizer(model_dir: str) -> AutoTokenizer:
  """Loads a Qwen3 HF tokenizer with pad set to eos if unset."""
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer
