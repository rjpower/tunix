"""Wrap the tunix sampler as an agent-loop ``model_fn(messages) -> response``.

The agent loop (`eval.agent_loop`) is policy-agnostic: it just needs a callable
that maps the running chat ``messages`` to the next assistant response string.
This module builds that callable from an SFT'd Qwen3 + its tokenizer, rendering
the messages in the exact Qwen3 ChatML the model was SFT'd on and decoding with
``top_p=1.0`` (the tunix Sampler decodes greedily unless ``top_p`` is set).
"""

from tunix.generate import sampler as sampler_lib

from mega_eval.training.agent_sft import render_chatml


def make_tunix_model_fn(
    model,
    tokenizer,
    mesh,
    *,
    max_prompt_length: int = 8192,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    seed: int = 0,
):
  """Returns ``model_fn(messages) -> response_text`` backed by the tunix sampler.

  Args:
    model: the SFT'd Qwen3 (loaded with remat=NONE so the KV-cache sampler works).
    tokenizer: the Qwen3 tokenizer.
    mesh: the device mesh the model is sharded on.
    max_prompt_length: max prompt tokens (agent prompts grow with terminal output).
    max_new_tokens: max tokens per action.
    temperature: sampling temperature (low for agentic determinism).
    seed: sampling seed.

  Returns:
    A callable suitable for :func:`eval.agent_loop.run_episode`.
  """
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  eos_id = int(tokenizer.eos_token_id)
  cache_config = sampler_lib.CacheConfig(
      cache_size=max_prompt_length + max_new_tokens + 16,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
  # Advance the seed every call so generations diverge — across turns AND across
  # repeated episodes (pass@k). A fixed seed makes every draw identical even at
  # temperature>0 (the tunix Sampler keys randomness on seed; top_p=1.0 is what
  # makes it honor temperature+seed at all). Deterministic sequence => reproducible.
  _state = {"seed": seed}

  def _fit_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trims history to <= max_prompt_length tokens, keeping msg[0] + recent turns.

    The agent context grows every turn (accumulated terminal output); the tunix
    Sampler uses the ACTUAL prompt length, so an over-long prompt blows the KV
    cache ("Total sampling steps N must be less than cache size"). Keep the first
    message (system + task) and drop the oldest following turns until it fits.
    """
    if len(messages) <= 1:
      return messages
    head, tail = messages[:1], messages[1:]
    while tail:
      candidate = head + tail
      n = len(tokenizer.encode(render_chatml(candidate, add_generation_prompt=True)))
      if n <= max_prompt_length:
        return candidate
      tail = tail[1:]  # drop the oldest non-head turn
    return head

  def model_fn(messages: list[dict[str, str]]) -> str:
    prompt = render_chatml(_fit_messages(messages), add_generation_prompt=True)
    call_seed = _state["seed"]
    _state["seed"] += 1
    with mesh:
      out = sampler(
          input_strings=[prompt],
          max_generation_steps=max_new_tokens,
          max_prompt_length=max_prompt_length,
          echo=False,
          eos_tokens=[im_end_id, eos_id],
          temperature=temperature,
          top_p=1.0,  # REQUIRED: tunix Sampler decodes greedily without top_p.
          seed=call_seed,
      )
    return out.text[0]

  return model_fn
