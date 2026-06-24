"""Grade a Terminal-Bench task after the agent has acted in its container.

Terminal-Bench grading contract (from each task's ``tests/test.sh``): copy the
``tests/`` tree into the container, run ``tests/test.sh`` which runs
``pytest /tests/test_outputs.py`` and writes a score to ``reward.txt``. We read
that score (fraction in [0, 1]); a task counts as solved at score >= 1.0. If no
``reward.txt`` is produced we fall back to the pytest exit code (0 == pass).
"""

import dataclasses
import re

from mega_eval.eval.sandbox import ExecResult, GvisorContainerSandbox
from mega_eval.eval.tb_tasks import TBTask

# Where tasks commonly write the verifier score (test.sh creates /logs/verifier).
_REWARD_PATHS = ("/logs/verifier/reward.txt", "/tests/reward.txt", "reward.txt", "/reward.txt")


@dataclasses.dataclass(frozen=True)
class GradeResult:
  """Outcome of grading one task."""

  task_id: str
  solved: bool
  score: float
  test_exit_code: int
  detail: str


def _parse_reward(text: str) -> float | None:
  m = re.search(r"[-+]?\d*\.?\d+", text)
  return float(m.group()) if m else None


def grade_task(sandbox: GvisorContainerSandbox, task: TBTask) -> GradeResult:
  """Copies the task's tests into ``sandbox`` and runs the grader.

  Args:
    sandbox: the gVisor container the agent acted in (filesystem state preserved).
    task: the TB task being graded.

  Returns:
    A :class:`GradeResult`.
  """
  cp = sandbox.copy_in(task.tests_dir, "/tests")
  if cp.exit_code != 0:
    return GradeResult(task.task_id, False, 0.0, cp.exit_code, f"copy tests failed: {cp.stderr}")

  run = sandbox.exec("bash /tests/test.sh", timeout=task.verifier_timeout_sec)

  score: float | None = None
  for path in _REWARD_PATHS:
    cat = sandbox.exec(f"cat {path} 2>/dev/null", timeout=15)
    if cat.exit_code == 0 and cat.stdout.strip():
      score = _parse_reward(cat.stdout)
      if score is not None:
        break

  if score is None:
    # No reward file -> use the grader's exit code.
    solved = run.exit_code == 0
    score = 1.0 if solved else 0.0
    detail = "graded by test.sh exit code (no reward.txt)"
  else:
    solved = score >= 1.0
    detail = f"reward.txt score={score}"

  return GradeResult(
      task_id=task.task_id,
      solved=solved,
      score=score,
      test_exit_code=run.exit_code,
      detail=detail,
  )
