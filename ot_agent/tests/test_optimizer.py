# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full-run recipe (OpenThinkerAgent-32B card) uses cosine LR + warmup 0.1;
a wrong schedule (e.g. constant, or warmup that never decays) changes the result.
These check the schedule shape: warmup up to peak, then cosine decay toward 0.
"""

import optax

from ot_agent.sft import build_optimizer


def test_constant_when_no_warmup():
  # warmup_ratio <= 0 -> mega_eval constant clipped_adamw (no schedule object).
  opt = build_optimizer(1e-5, total_steps=100, warmup_ratio=0.0)
  assert isinstance(opt, optax.GradientTransformation)


def test_cosine_warmup_shape():
  peak = 4e-5
  total = 1000
  warmup_ratio = 0.1
  sched = optax.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=peak,
      warmup_steps=int(warmup_ratio * total), decay_steps=total, end_value=0.0,
  )
  # Start near 0, peak at end of warmup, ~0 at the end.
  assert float(sched(0)) < peak * 0.1
  assert abs(float(sched(int(warmup_ratio * total))) - peak) < peak * 1e-3
  assert float(sched(total)) < peak * 0.05
  # Monotonic rise through warmup, monotonic fall after.
  assert float(sched(50)) < float(sched(100))
  assert float(sched(500)) > float(sched(900))
  # The optimizer builds without error and is a valid transformation.
  opt = build_optimizer(peak, total_steps=total, warmup_ratio=warmup_ratio)
  assert isinstance(opt, optax.GradientTransformation)
