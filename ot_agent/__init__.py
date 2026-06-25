# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenThoughts-Agent (OT-Agent) post-training: Qwen3-32B SFT on 32x H100.

Replicates the SFT stage of *OpenThoughts-Agent* (arXiv:2606.24855): fine-tune
``Qwen/Qwen3-32B`` on the released 100K agent-trajectory set
(``open-thoughts/OpenThoughts-Agent-SFT-100K``) and, later, an RL pass on top.

This package is the **GPU multi-node** counterpart to the in-repo ``mega_eval``
project (which proved the same recipe on Qwen3-8B / TPU). It reuses ``mega_eval``
as a library -- the ChatML assistant-turn-masked encoder
(``mega_eval.training.agent_sft``), the Qwen3 loader (``mega_eval.models``), and
the clipped-AdamW / mesh / metrics glue (``mega_eval.training.common``) -- and
adds only what 32B-on-4x8-H100 needs:

  * ``distributed`` -- multi-node JAX rendezvous (one mesh across 4 nodes) via the
    iris endpoint registry (``iris.runtime.jax_init.initialize_jax``).
  * ``data``        -- the OT-Agent SFT scaling ladder (1K..100K) with
    **process-disjoint** sharding so a 4-process data-parallel run sees each
    example once per epoch (the ``mega_eval`` single-process pipeline does not).
  * ``sft``         -- the multi-process SFT driver (PeftTrainer + checkpoint).
  * ``launch_sft``  -- the env-configured iris entrypoint.
  * ``submit_sft.sh`` -- the multi-node submit (``iris job run --replicas N``,
    which auto-enables ``leafgroup`` coscheduling for multi-node GPU).
  * ``export_hf``   -- gather the sharded actor and write a single HF-format
    safetensors checkpoint (mirrored to R2), the coherent downstream artifact.
"""
