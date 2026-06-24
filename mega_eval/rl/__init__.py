# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dr.GRPO RL stage for the OpenThoughts terminal agent (tunix agentic learner).

The terminal agent is multi-turn, so RL uses tunix's *agentic* GRPO learner
(`tunix.rl.agentic`), which drives an (agent, environment) pair per rollout:

  * :class:`rl.agent.TerminusAgent` parses each model turn's Terminus-2 JSON
    action and hands the environment the shell to run -- reusing the exact eval
    loop logic so RL rollouts stay in the SFT distribution.
  * :class:`rl.environment.TerminalBenchEnv` runs that shell in the gVisor
    sandbox and, at episode end, grades the container -> a sparse reward in
    [0, 1]. With ``reward_fns=None`` the agentic reward manager uses this env
    reward directly, and Dr.GRPO (``advantage_estimator="drgrpo"``) turns the
    per-group reward spread into advantages.

See ``launch_rl.py`` for the RLCluster wiring and the iris submit command.
"""
