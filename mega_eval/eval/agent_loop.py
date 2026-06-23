"""The Terminus-2 agent loop: model -> JSON action -> sandbox -> observation.

This mirrors the harness the OpenThoughts agent SFT traces were generated with.
Each assistant turn is a JSON object::

    {
      "analysis": "...",
      "plan": "...",
      "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}, ...]
    }

The model is sometimes chatty (prose before the JSON), so :func:`parse_action`
extracts the last balanced JSON object that has a ``commands`` key. We execute the
concatenated ``keystrokes`` in the sandbox and feed the terminal output back as a
``user`` turn ("Current terminal state: New Terminal Output: ..."), exactly as the
training data is shaped, so the SFT'd policy sees an in-distribution context.

This module is model- and sandbox-agnostic: pass a ``model_fn(messages) -> str``
and any :class:`~eval.sandbox.Sandbox`. That lets the loop be unit-tested on CPU
with a scripted model + the local sandbox, while production wires the tunix
sampler + the gVisor sandbox.
"""

import dataclasses
import json
from typing import Any, Callable

from mega_eval.eval.sandbox import Sandbox

# The Terminus-2 system prompt (verbatim from the SFT traces' turn 0 preamble), so
# the eval prompt matches the training distribution.
SYSTEM_PROMPT = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux "
    "environment. You will be given a task description and the output from "
    "previously executed commands. Your goal is to solve the task by providing "
    "batches of shell commands.\n\n"
    "Format your response as JSON with the following structure:\n\n"
    "{\n"
    '  "analysis": "Analyze the current state based on the terminal output '
    'provided. What do you see? What has been accomplished? What still needs to '
    'be done?",\n'
    '  "plan": "Describe your plan for the next steps. What commands will you run '
    'and why?",\n'
    '  "commands": [\n'
    '    {\n'
    '      "keystrokes": "ls -la\\n",\n'
    '      "duration": 0.1\n'
    "    }\n"
    "  ],\n"
    '  "task_complete": false\n'
    "}\n\n"
    "Set \"task_complete\" to true only when you are confident the task is fully "
    "solved. Provide an empty commands list when complete."
)

ModelFn = Callable[[list[dict[str, str]]], str]


@dataclasses.dataclass(frozen=True)
class Action:
  """A parsed agent action."""

  analysis: str
  plan: str
  commands: list[dict[str, Any]]
  task_complete: bool
  raw: str
  parse_ok: bool


def _iter_json_objects(text: str):
  """Yields substrings of ``text`` that are balanced ``{...}`` spans (brace-matched).

  Naive but robust to prose around the JSON: tracks brace depth while respecting
  string literals and escapes. Yields candidate spans in order of appearance.
  """
  depth = 0
  start = -1
  in_str = False
  escape = False
  for i, ch in enumerate(text):
    if in_str:
      if escape:
        escape = False
      elif ch == "\\":
        escape = True
      elif ch == '"':
        in_str = False
      continue
    if ch == '"':
      in_str = True
    elif ch == "{":
      if depth == 0:
        start = i
      depth += 1
    elif ch == "}":
      if depth > 0:
        depth -= 1
        if depth == 0 and start >= 0:
          yield text[start : i + 1]


def parse_action(response: str) -> Action:
  """Extracts the agent action from a model response.

  Picks the LAST balanced JSON object that has a ``commands`` key (the action),
  tolerating leading prose / multiple objects. Falls back to ``parse_ok=False``
  with empty commands if none parses.
  """
  best: dict[str, Any] | None = None
  for span in _iter_json_objects(response):
    try:
      obj = json.loads(span)
    except (json.JSONDecodeError, ValueError):
      continue
    if isinstance(obj, dict) and "commands" in obj:
      best = obj  # keep the last valid action-shaped object
  if best is None:
    return Action("", "", [], False, response, parse_ok=False)
  cmds = best.get("commands") or []
  if not isinstance(cmds, list):
    cmds = []
  return Action(
      analysis=str(best.get("analysis", "")),
      plan=str(best.get("plan", "")),
      commands=[c for c in cmds if isinstance(c, dict) and "keystrokes" in c],
      task_complete=bool(best.get("task_complete", False)),
      raw=response,
      parse_ok=True,
  )


def _commands_to_shell(commands: list[dict[str, Any]]) -> str:
  """Concatenates command keystrokes into a single shell snippet.

  Keystrokes are literal terminal input (usually a full command ending in
  ``\\n``). We approximate the tmux session by running them as one bash script;
  this covers the non-interactive batch tasks that dominate Terminal-Bench.
  """
  return "".join(str(c.get("keystrokes", "")) for c in commands)


def format_observation(exec_stdout: str, exec_stderr: str, *, max_chars: int = 4000) -> str:
  """Builds the user-turn observation fed back to the model, training-data shaped."""
  out = exec_stdout
  if exec_stderr.strip():
    out = f"{out}\n[stderr]\n{exec_stderr}"
  if len(out) > max_chars:  # keep the tail (most recent output)
    out = "...(truncated)...\n" + out[-max_chars:]
  return f"Current terminal state:\nNew Terminal Output:\n{out}"


@dataclasses.dataclass
class EpisodeResult:
  """Outcome of one agent episode."""

  messages: list[dict[str, str]]
  turns: int
  completed: bool
  parse_failures: int


def run_episode(
    model_fn: ModelFn,
    sandbox: Sandbox,
    instruction: str,
    *,
    max_turns: int = 20,
    command_timeout: float = 60.0,
    system_prompt: str = SYSTEM_PROMPT,
) -> EpisodeResult:
  """Runs the agent on ``instruction`` against ``sandbox`` until done / max turns.

  Args:
    model_fn: ``messages -> response_text`` (the policy).
    sandbox: where commands execute (gVisor in prod, local in tests).
    instruction: the task instruction (from ``instruction.md``).
    max_turns: cap on agent turns.
    command_timeout: per-command-batch execution timeout.
    system_prompt: the Terminus-2 preamble.

  Returns:
    An :class:`EpisodeResult` with the full transcript.
  """
  messages: list[dict[str, str]] = [
      {"role": "user", "content": f"{system_prompt}\n\n# Task\n{instruction}"}
  ]
  parse_failures = 0
  completed = False
  turn = 0
  for turn in range(1, max_turns + 1):
    response = model_fn(messages)
    messages.append({"role": "assistant", "content": response})
    action = parse_action(response)
    if not action.parse_ok:
      parse_failures += 1
      messages.append({
          "role": "user",
          "content": (
              "Your response could not be parsed as a JSON action. Respond with a "
              "single JSON object containing analysis, plan, and commands."
          ),
      })
      continue
    if action.task_complete or not action.commands:
      completed = action.task_complete
      break
    shell = _commands_to_shell(action.commands)
    res = sandbox.exec(shell, timeout=command_timeout)
    messages.append({"role": "user", "content": format_observation(res.stdout, res.stderr)})
  return EpisodeResult(messages=messages, turns=turn, completed=completed, parse_failures=parse_failures)
