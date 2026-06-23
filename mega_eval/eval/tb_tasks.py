"""Load OpenThoughts Terminal-Bench (TB-dev) tasks from HuggingFace.

``open-thoughts/OpenThoughts-TB-dev`` ships each task as a directory::

    <task_id>/
      task.toml             # agent/verifier timeouts, metadata
      instruction.md        # the task prompt given to the agent
      environment/Dockerfile  # builds the task's container image
      environment/...       # files copied into the image
      solution/solve.sh     # reference solution (oracle; not given to the agent)
      tests/test.sh         # grader entrypoint: runs pytest -> writes reward
      tests/...             # grader files

The agent works inside the built environment image (a WORKDIR like ``/workdir``),
issues shell commands, then ``tests/`` is copied in and ``tests/test.sh`` decides
pass/fail.
"""

import dataclasses
import glob
import os

try:
  import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
  import tomli as tomllib  # type: ignore

from huggingface_hub import snapshot_download

DATASET_ID = "open-thoughts/OpenThoughts-TB-dev"
DATASET_REVISION = "0d54f719f34dca712c8d6ef0f51df4670a2a287a"


@dataclasses.dataclass(frozen=True)
class TBTask:
  """One Terminal-Bench task on local disk."""

  task_id: str
  root: str
  instruction: str
  agent_timeout_sec: float
  verifier_timeout_sec: float
  image_tag: str  # the docker tag we build the environment into

  @property
  def environment_dir(self) -> str:
    return os.path.join(self.root, "environment")

  @property
  def tests_dir(self) -> str:
    return os.path.join(self.root, "tests")


def download_tb_dev(*, revision: str = DATASET_REVISION) -> str:
  """Downloads the TB-dev task tree and returns its local root dir."""
  return snapshot_download(
      repo_id=DATASET_ID, repo_type="dataset", revision=revision
  )


def _load_one(root: str, task_id: str) -> TBTask | None:
  task_root = os.path.join(root, task_id)
  toml_path = os.path.join(task_root, "task.toml")
  instr_path = os.path.join(task_root, "instruction.md")
  if not (os.path.isfile(toml_path) and os.path.isfile(instr_path)):
    return None
  with open(toml_path, "rb") as f:
    meta = tomllib.load(f)
  with open(instr_path, "r") as f:
    instruction = f.read()
  agent_to = float(meta.get("agent", {}).get("timeout_sec", 1800.0))
  ver_to = float(meta.get("verifier", {}).get("timeout_sec", 600.0))
  return TBTask(
      task_id=task_id,
      root=task_root,
      instruction=instruction,
      agent_timeout_sec=agent_to,
      verifier_timeout_sec=ver_to,
      image_tag=f"ota-tb/{task_id.lower()}:latest",
  )


def load_tb_tasks(
    *, revision: str = DATASET_REVISION, limit: int | None = None
) -> list[TBTask]:
  """Loads TB-dev tasks (each a directory with task.toml + instruction.md)."""
  root = download_tb_dev(revision=revision)
  task_ids = sorted(
      d for d in os.listdir(root)
      if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
      and os.path.isfile(os.path.join(root, d, "task.toml"))
  )
  tasks = []
  for tid in task_ids:
    t = _load_one(root, tid)
    if t is not None:
      tasks.append(t)
    if limit is not None and len(tasks) >= limit:
      break
  return tasks
