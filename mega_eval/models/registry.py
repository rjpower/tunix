"""Registry mapping a model name to its HF repo + tunix loaders.

The OpenThoughts terminal-agent recipe (https://www.openthoughts.ai/blog/agent)
post-trains ``Qwen/Qwen3-8B`` (the released chat model, which already ships the
ChatML template the agent traces are formatted with). ``qwen3-8b-base`` is
provided for an ablation that starts from the raw base LM instead.

All loaders share ``load(model_dir, *, mesh=None, dtype=..., param_dtype=...,
remat=..., use_flash_attention=...) -> nnx model`` and ``load_tokenizer(model_dir)
-> HF tokenizer``. The eos id is taken from ``tokenizer.eos_token_id``.
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """A base model: its display name, HF repo, and tunix loaders."""

  name: str
  repo: str
  load_model: Callable  # (model_dir, *, mesh=None, dtype=..., ...) -> nnx model
  load_tokenizer: Callable  # (model_dir) -> HF tokenizer


def _qwen3_8b_spec() -> ModelSpec:
  from mega_eval.models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-8b",
      repo="Qwen/Qwen3-8B",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


def _qwen3_8b_base_spec() -> ModelSpec:
  from mega_eval.models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-8b-base",
      repo="Qwen/Qwen3-8B-Base",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


def _qwen3_1_7b_base_spec() -> ModelSpec:
  # Small control arm for fast CPU/v6e-4 smoke tests of the SFT plumbing.
  from mega_eval.models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-1.7b-base",
      repo="Qwen/Qwen3-1.7B-Base",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


def _qwen3_32b_spec() -> ModelSpec:
  # The OpenThoughts-Agent paper (arXiv:2606.24855) SFTs ``Qwen/Qwen3-32B`` on
  # the 100K agent set; this is the ``ot_agent`` replication target on 4x8 H100.
  # The generic ``load_qwen3`` loader reads ``config.json`` and matches tunix's
  # built-in ``ModelConfig.qwen3_32b`` preset (64 layers, embed 5120, no
  # rope_scaling, untied embeddings) -- no model-code change for 32B.
  from mega_eval.models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-32b",
      repo="Qwen/Qwen3-32B",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


def _qwen3_32b_base_spec() -> ModelSpec:
  # Base-model ablation arm (raw LM rather than the released chat model).
  from mega_eval.models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-32b-base",
      repo="Qwen/Qwen3-32B-Base",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


# Name -> factory (lazy so importing this module doesn't import the loader).
_REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "qwen3-8b": _qwen3_8b_spec,
    "qwen3-8b-base": _qwen3_8b_base_spec,
    "qwen3-1.7b-base": _qwen3_1_7b_base_spec,
    "qwen3-32b": _qwen3_32b_spec,
    "qwen3-32b-base": _qwen3_32b_base_spec,
}


def get_model_spec(name: str = "qwen3-8b") -> ModelSpec:
  """Returns the :class:`ModelSpec` for ``name`` (default ``qwen3-8b``).

  Raises:
    KeyError: if ``name`` is not registered.
  """
  key = name.strip().lower()
  if key not in _REGISTRY:
    raise KeyError(f"Unknown model {name!r}; known: {sorted(_REGISTRY)}.")
  return _REGISTRY[key]()
