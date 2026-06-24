"""CPU tests for the Terminus-2 agent loop: action parsing + the run loop.

Uses a scripted model + the local (unsafe) sandbox, so it runs anywhere with no
weights/accelerator/gVisor.
"""

from mega_eval.eval.agent_loop import parse_action, run_episode
from mega_eval.eval.sandbox import LocalUnsafeSandbox


def test_parse_pure_json():
  a = parse_action('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}')
  assert a.parse_ok
  assert a.analysis == "a" and a.plan == "p"
  assert len(a.commands) == 1 and a.commands[0]["keystrokes"] == "ls\n"
  assert not a.task_complete


def test_parse_prose_then_json():
  resp = 'Let me analyze what happened:\n1. dir exists\n\n{"analysis": "x", "commands": [{"keystrokes": "pwd\\n"}]}'
  a = parse_action(resp)
  assert a.parse_ok and a.commands[0]["keystrokes"] == "pwd\n"


def test_parse_last_action_object_wins():
  # Two action-shaped objects; the last (final) one is the real action.
  resp = '{"commands": [{"keystrokes": "echo old\\n"}]} ... {"commands": [{"keystrokes": "echo new\\n"}]}'
  a = parse_action(resp)
  assert a.parse_ok and a.commands[0]["keystrokes"] == "echo new\n"


def test_parse_ignores_non_action_json():
  # A JSON object without "commands" is not an action.
  a = parse_action('{"note": "no commands here"}')
  assert not a.parse_ok and a.commands == []


def test_parse_handles_braces_in_strings():
  a = parse_action('{"analysis": "use awk \'{print $1}\'", "commands": [{"keystrokes": "x\\n"}]}')
  assert a.parse_ok and a.analysis == "use awk '{print $1}'"


def test_parse_failure_on_garbage():
  a = parse_action("I cannot help with that.")
  assert not a.parse_ok


def test_run_episode_executes_and_completes():
  # Turn 1: write a file. Turn 2: declare complete.
  scripted = [
      '{"analysis": "create file", "plan": "echo", "commands": [{"keystrokes": "echo hi > /tmp/ota_test_file\\n"}]}',
      '{"analysis": "done", "plan": "stop", "commands": [], "task_complete": true}',
  ]
  calls = {"i": 0}

  def model_fn(messages):
    # The observation from turn 1 must be visible to the model on turn 2.
    if calls["i"] == 1:
      assert any("Current terminal state" in m["content"] for m in messages)
    r = scripted[calls["i"]]
    calls["i"] += 1
    return r

  sb = LocalUnsafeSandbox()
  result = run_episode(model_fn, sb, "make a file", max_turns=5)
  sb.close()
  assert result.completed
  assert result.turns == 2
  assert result.parse_failures == 0


def test_run_episode_handles_parse_failure_then_recovers():
  scripted = [
      "garbage no json",
      '{"analysis": "ok", "commands": [], "task_complete": true}',
  ]
  calls = {"i": 0}

  def model_fn(messages):
    r = scripted[calls["i"]]
    calls["i"] += 1
    return r

  sb = LocalUnsafeSandbox()
  result = run_episode(model_fn, sb, "task", max_turns=5)
  sb.close()
  assert result.parse_failures == 1 and result.completed
