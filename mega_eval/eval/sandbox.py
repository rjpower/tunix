"""Isolated command execution for agent tool use, via gVisor (runsc).

The OpenThoughts terminal agent issues shell commands that we must NOT run on the
training host. Terminal-Bench tasks ship their own Docker environment, so the
sandbox model is: run each task's image as a container under the **runsc** OCI
runtime (gVisor), and ``docker exec`` the agent's commands into it.

This runs inside the iris TPU task container, which is ``--privileged`` (iris adds
that for accelerators), so rootful gVisor + a task-local dockerd work. The custom
task image (`docker/Dockerfile.agent-task`) ships `runsc`, `docker`, and a
`/etc/docker/daemon.json` registering the `runsc` runtime.

Implementations:
  * :class:`GvisorContainerSandbox` -- a long-lived container per task, started
    with ``docker run --runtime=runsc``; the production path.
  * :class:`LocalUnsafeSandbox` -- plain subprocess, NO isolation. For developing
    the agent-loop / grading logic on a laptop only; never use on untrusted code.

Use :func:`make_sandbox` to pick by ``OTA_SANDBOX`` env (``gvisor`` | ``local``).
"""

import dataclasses
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import uuid
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class ExecResult:
  """Result of one command execution."""

  stdout: str
  stderr: str
  exit_code: int
  timed_out: bool = False


class Sandbox(Protocol):
  """A place to run agent shell commands in isolation."""

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    """Runs ``command`` (a shell string) and returns its result."""
    ...

  def close(self) -> None:
    """Tears down any container / resources."""
    ...


def _as_text(value) -> str:
  """Coerces subprocess output (str | bytes | None) to str, never raising.

  Agent commands run arbitrary programs whose stdout/stderr may be non-UTF-8
  (binary dumps, latin-1 logs), so a strict decode would crash the episode/rollout
  on a stray byte. And on a timeout ``TimeoutExpired.stdout/stderr`` come back as
  *bytes* even when ``text=True`` was requested, which then breaks a ``str +``
  concat. Decoding with ``errors="replace"`` handles both.
  """
  if value is None:
    return ""
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return value


def _run(argv: list[str], *, timeout: float, input_text: str | None = None) -> ExecResult:
  """Runs a subprocess with a hard timeout, capturing stdout/stderr.

  Output is decoded leniently (``errors="replace"``) because agent-issued commands
  can emit non-UTF-8 bytes; a strict decode would raise mid-episode. See
  :func:`_as_text` for the timeout-path bytes handling.
  """
  try:
    proc = subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    return ExecResult(_as_text(proc.stdout), _as_text(proc.stderr), proc.returncode)
  except subprocess.TimeoutExpired as e:
    return ExecResult(_as_text(e.stdout), _as_text(e.stderr) + "\n[timeout]", 124, timed_out=True)


# --- Runtime bootstrap of docker + runsc -------------------------------------
# Two ways to get the sandbox runtime into the privileged iris task:
#   * the custom task image (docker/Dockerfile.agent-task) bakes these in -- then
#     ensure_sandbox_runtime() is a pure no-op; OR
#   * on the stock iris image, we download the exact same binaries at runtime.
# This lets the eval run either way, so we don't hard-depend on the custom image
# being published. Confirmed working on a v6e TPU task (see eval/gvisor_smoke.py).
_DOCKER_VERSION = "27.3.1"
_DOCKER_TGZ_URL = f"https://download.docker.com/linux/static/stable/x86_64/docker-{_DOCKER_VERSION}.tgz"
# Docker 27's `docker build` requires BuildKit (the legacy builder was removed),
# and the static docker tarball does NOT bundle the buildx CLI plugin -- without
# it builds fail with "unable to open Dockerfile". Install it separately.
_BUILDX_VERSION = "0.17.1"
_BUILDX_URL = f"https://github.com/docker/buildx/releases/download/v{_BUILDX_VERSION}/buildx-v{_BUILDX_VERSION}.linux-amd64"
_CLI_PLUGINS_DIR = "/usr/local/lib/docker/cli-plugins"
_RUNSC_URL = "https://storage.googleapis.com/gvisor/releases/release/latest/x86_64/runsc"
_BIN_DIR = "/usr/local/bin"
_DAEMON_JSON_PATH = "/etc/docker/daemon.json"
# ptrace: no /dev/kvm in iris tasks. ignore-cgroups: the task cgroup is restricted
# so runsc can't write cgroup.subtree_control. network=sandbox: gVisor netstack.
_RUNSC_RUNTIME_ARGS = ["--platform=ptrace", "--network=sandbox", "--ignore-cgroups"]

# dockerd flags for running nested inside a container: vfs avoids nested-overlayfs
# failures, and we disable the bridge/iptables since sandbox containers use
# ``--network none`` (no docker bridge needed). Override via ``DOCKERD_ARGS``.
_DEFAULT_DOCKERD_ARGS = ["--storage-driver=vfs", "--iptables=false", "--bridge=none"]
_DOCKERD_LOG = "/tmp/dockerd.out"


def _install_static_docker() -> None:
  urllib.request.urlretrieve(_DOCKER_TGZ_URL, "/tmp/ota-docker.tgz")
  with tarfile.open("/tmp/ota-docker.tgz") as t:
    t.extractall("/tmp/ota-docker-extract")
  src = "/tmp/ota-docker-extract/docker"
  for fn in os.listdir(src):
    dst = os.path.join(_BIN_DIR, fn)
    shutil.copy(os.path.join(src, fn), dst)
    os.chmod(dst, 0o755)


def _install_runsc() -> None:
  dst = os.path.join(_BIN_DIR, "runsc")
  urllib.request.urlretrieve(_RUNSC_URL, dst)
  os.chmod(dst, 0o755)


def _install_buildx() -> None:
  os.makedirs(_CLI_PLUGINS_DIR, exist_ok=True)
  dst = os.path.join(_CLI_PLUGINS_DIR, "docker-buildx")
  urllib.request.urlretrieve(_BUILDX_URL, dst)
  os.chmod(dst, 0o755)


def _ensure_runsc_runtime_registered() -> None:
  """Reconciles the runsc runtime in daemon.json to our canonical definition.

  Always overwrites the ``runsc`` entry (not just when absent): a custom task
  image may ship a STALE daemon.json -- e.g. one missing ``--ignore-cgroups`` --
  and silently skipping the update would leave runsc unable to start containers
  ("cannot set up cgroup for root"). dockerd reads daemon.json at startup and
  :func:`ensure_dockerd` runs this before starting it, so the corrected args take
  effect. Idempotent: no write if already canonical.
  """
  existing: dict = {}
  if os.path.exists(_DAEMON_JSON_PATH):
    try:
      with open(_DAEMON_JSON_PATH) as f:
        existing = json.load(f)
    except (OSError, json.JSONDecodeError):
      existing = {}
  desired = {"path": os.path.join(_BIN_DIR, "runsc"), "runtimeArgs": _RUNSC_RUNTIME_ARGS}
  runtimes = existing.get("runtimes", {})
  if runtimes.get("runsc") == desired:
    return  # already canonical
  runtimes["runsc"] = desired
  existing["runtimes"] = runtimes
  os.makedirs(os.path.dirname(_DAEMON_JSON_PATH), exist_ok=True)
  with open(_DAEMON_JSON_PATH, "w") as f:
    json.dump(existing, f)


def ensure_sandbox_runtime() -> None:
  """Idempotently ensure docker + runsc + the runsc Docker runtime are available.

  No-op on the custom openthoughts-agent-task image (everything is pre-baked); on
  the stock iris image it downloads Docker's static binaries + runsc at runtime.
  Must run before dockerd starts, since dockerd reads daemon.json at startup.
  """
  if shutil.which("docker") is None:
    _install_static_docker()
  if shutil.which("runsc") is None:
    _install_runsc()
  if not any(
      os.path.isfile(os.path.join(d, "docker-buildx"))
      for d in (_CLI_PLUGINS_DIR, os.path.expanduser("~/.docker/cli-plugins"))
  ):
    _install_buildx()
  _ensure_runsc_runtime_registered()


def ensure_dockerd(*, timeout: float = 120.0) -> None:
  """Bootstraps the runtime if needed, starts a task-local dockerd, and waits.

  Idempotent. Requires the ``--privileged`` task container (TPU jobs have it).
  Runs dockerd with vfs storage + no bridge/iptables, which is what lets it come
  up nested inside the iris task container.

  Raises:
    RuntimeError: if dockerd does not become ready within ``timeout`` (the error
      includes the tail of dockerd's own log to make the cause visible).
  """
  ensure_sandbox_runtime()
  if _run(["docker", "info"], timeout=10).exit_code == 0:
    return
  extra = os.environ.get("DOCKERD_ARGS")
  args = extra.split() if extra else _DEFAULT_DOCKERD_ARGS
  subprocess.Popen(
      ["dockerd", *args],
      stdout=open(_DOCKERD_LOG, "w"),
      stderr=subprocess.STDOUT,
  )
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if _run(["docker", "info"], timeout=10).exit_code == 0:
      return
    time.sleep(1.0)
  tail = ""
  try:
    with open(_DOCKERD_LOG) as f:
      tail = "".join(f.readlines()[-25:])
  except OSError:
    pass
  raise RuntimeError(f"dockerd not ready after {timeout}s. dockerd log tail:\n{tail}")


class GvisorContainerSandbox:
  """A Terminal-Bench task environment running under gVisor.

  Starts ``image`` as a detached container with the ``runsc`` runtime and execs
  agent commands into it via ``docker exec``. The container is removed on
  :meth:`close`.
  """

  def __init__(
      self,
      image: str,
      *,
      workdir: str | None = None,
      runtime: str = "runsc",
      network: str = "none",
      name: str | None = None,
      mem_limit: str = "4g",
      cpus: str = "2",
  ):
    ensure_dockerd()
    self.image = image
    self.workdir = workdir  # None => use the image's own WORKDIR
    # uuid (not pid+time): RL rollouts boot G containers concurrently in one
    # process, and a pid+millisecond name collides ("container name already in use").
    self._name = name or f"ota-task-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    argv = [
        "docker", "run", "-d", "--rm",
        "--runtime", runtime,
        "--network", network,
        "--memory", mem_limit,
        "--cpus", cpus,
    ]
    if workdir:
      argv += ["--workdir", workdir]
    # Keep the container alive with a no-op PID 1 so we can exec into it.
    argv += ["--name", self._name, image, "sleep", "infinity"]
    res = _run(argv, timeout=300)
    if res.exit_code != 0:
      raise RuntimeError(
          f"failed to start sandbox container from {image!r}: {res.stderr}"
      )

  @property
  def name(self) -> str:
    """The container name (for ``docker cp`` / grading)."""
    return self._name

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    argv = ["docker", "exec"]
    if self.workdir:
      argv += ["--workdir", self.workdir]
    argv += [self._name, "bash", "-lc", command]
    return _run(argv, timeout=timeout)

  def copy_in(self, local_path: str, container_path: str, *, timeout: float = 120.0) -> ExecResult:
    """Copies a host path into the container (``docker cp``), dereferencing symlinks.

    HF ``snapshot_download`` stores files as symlinks into a ``blobs/`` cache; a
    plain ``docker cp`` would copy them as symlinks that dangle inside the
    container (the blob path doesn't exist there). So we stage a dereferenced
    copy first -- ``shutil.copy`` follows file symlinks; ``copytree(symlinks=
    False)`` follows them through a tree.

    For a single file we ``docker cp`` it INTO the destination's parent dir
    rather than to the exact dest path: a single-file ``docker cp`` to a
    not-yet-existing dest path misbehaves under the runsc runtime (it lands as a
    broken/empty target), whereas copying into an existing dir works (as the
    directory path does).
    """
    staging = tempfile.mkdtemp(prefix="ota-cp-")
    try:
      if os.path.isdir(local_path):
        src = os.path.join(staging, os.path.basename(local_path.rstrip("/")) or "payload")
        shutil.copytree(
            src=local_path, dst=src, symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        return _run(["docker", "cp", src, f"{self._name}:{container_path}"], timeout=timeout)
      # Single file: stage under the DEST basename, cp into the dest's parent dir.
      dst_dir = os.path.dirname(container_path.rstrip("/")) or "/"
      src = os.path.join(staging, os.path.basename(container_path.rstrip("/")))
      shutil.copy(local_path, src)
      return _run(["docker", "cp", src, f"{self._name}:{dst_dir}/"], timeout=timeout)
    finally:
      shutil.rmtree(staging, ignore_errors=True)

  def close(self) -> None:
    _run(["docker", "rm", "-f", self._name], timeout=30)


# --- Optional Docker Hub pull-through mirror (Artifact Registry) -------------
# Every TB-dev task Dockerfile pulls a Docker Hub *official* base (ubuntu:*,
# python:*-slim*). Under a wide eval fan-out that risks Docker Hub rate limits,
# so we optionally redirect those pulls through an Artifact Registry remote repo
# (pull-through cache) by rewriting the staged Dockerfile's FROM lines.
#
# Set DOCKER_REGISTRY_MIRROR to the AR remote-repo prefix, e.g.
#   us-docker.pkg.dev/hai-gcp-models/docker-mirror      (US jobs)
#   europe-docker.pkg.dev/hai-gcp-models/docker-mirror  (EU jobs)
# Unset => no rewrite (pull straight from Docker Hub). Only docker.io images are
# mirrored; refs to other registries (gcr.io/ghcr.io/quay.io/...) and `scratch`
# are left untouched. dockerd's registry-mirrors can't auth to a private AR repo,
# so rewriting the staged FROM lines + a plain `docker login` is the robust path.
_MIRROR_ENV = "DOCKER_REGISTRY_MIRROR"
_FROM_RE = re.compile(r"^(\s*FROM\s+)((?:--\S+\s+)*)(\S+)(.*)$", re.IGNORECASE)
_ar_logged_in: set[str] = set()


def _mirror_image_ref(ref: str, mirror_prefix: str) -> str | None:
  """Maps a Docker Hub image ref to the AR mirror path, or None to leave as-is.

  Docker Hub *official* images (single-component names like ``ubuntu``) live
  under ``library/`` in the registry, so the mirror path needs that prefix; an
  ``org/name`` ref keeps its namespace. Refs to a non-Docker-Hub registry (a
  first path component containing ``.``/``:``, e.g. ``gcr.io/...``) and
  ``scratch`` return None (a Docker Hub remote repo can't serve them).
  """
  if ref.lower() == "scratch":
    return None
  body = ref
  if body.startswith("docker.io/"):
    body = body[len("docker.io/"):]
  else:
    first = body.split("/", 1)[0]
    if "/" in body and ("." in first or ":" in first or first == "localhost"):
      return None  # some other registry — our docker-mirror can't serve it
  # `body` is now a Docker Hub repo path (+ optional :tag / @digest).
  name = re.split(r"[:@]", body, maxsplit=1)[0]
  if "/" not in name:
    body = "library/" + body  # official image lives under library/
  return mirror_prefix.rstrip("/") + "/" + body


def _rewrite_dockerfile_for_mirror(text: str, mirror_prefix: str) -> str:
  """Rewrites each FROM line's image ref to the AR mirror (where applicable).

  Preserves ``--platform`` flags and any ``AS <stage>`` suffix; leaves non-FROM
  lines, ``scratch``, intra-build stage refs, and non-Docker-Hub registries
  untouched.
  """
  out = []
  for line in text.splitlines():
    m = _FROM_RE.match(line)
    if m:
      pre, flags, image, rest = m.groups()
      new = _mirror_image_ref(image, mirror_prefix)
      if new is not None:
        line = f"{pre}{flags}{new}{rest}"
    out.append(line)
  return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _ensure_ar_login(mirror_prefix: str) -> None:
  """``docker login`` to the AR host of ``mirror_prefix`` via the GCE metadata token.

  AR remote repos are private, so the nested dockerd needs creds to pull through
  them. The TPU VM's service account already has artifactregistry.reader (iris
  pulls the task base image through the sibling ghcr mirror), so we mint a
  short-lived OAuth token from the metadata server -- no gcloud needed. Cached
  per host so repeated builds log in once.
  """
  host = mirror_prefix.split("/", 1)[0]
  if host in _ar_logged_in:
    return
  url = ("http://metadata.google.internal/computeMetadata/v1/instance/"
         "service-accounts/default/token")
  req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
  with urllib.request.urlopen(req, timeout=10) as r:
    token = json.load(r)["access_token"]
  res = _run(["docker", "login", "-u", "oauth2accesstoken", "-p", token, host], timeout=30)
  if res.exit_code != 0:
    raise RuntimeError(f"docker login to AR mirror {host!r} failed: {res.stderr}")
  _ar_logged_in.add(host)


def build_image(context_dir: str, tag: str, *, timeout: float = 1200.0) -> ExecResult:
  """Builds a Docker image from ``context_dir`` (must contain a Dockerfile).

  Uses ``docker buildx build --load`` (BuildKit): Docker 27 dropped the legacy
  builder, so a plain ``docker build`` fails without the buildx plugin.
  ``--load`` writes the result into the local image store so ``docker run`` (and
  hence the gVisor sandbox) can use it.

  HuggingFace ``snapshot_download`` stores files as symlinks into a ``blobs/``
  cache; BuildKit can't follow symlinks that point outside the build context, so
  we first materialize the context with symlinks dereferenced.

  If ``DOCKER_REGISTRY_MIRROR`` is set, the staged Dockerfile's Docker Hub FROM
  lines are rewritten to pull through that Artifact Registry remote repo (and we
  ``docker login`` to it once) -- a pull-through cache that avoids Docker Hub
  rate limits under a wide fan-out. Unset => pull straight from Docker Hub.

  Build RUN steps (apt-get, etc.) need network egress, but dockerd runs with
  ``--bridge=none`` (so it can come up nested without iptables, which the stock
  iris image lacks). So we build with ``--network=host`` -- RUN steps use the
  task's own network. The sandboxed run-time containers still use ``--network
  none``.
  """
  ensure_dockerd()
  mirror = os.environ.get(_MIRROR_ENV, "").strip()
  staging = tempfile.mkdtemp(prefix="ota-build-")
  ctx = os.path.join(staging, "context")
  try:
    shutil.copytree(
        context_dir, ctx, symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    dockerfile = os.path.join(ctx, "Dockerfile")
    if mirror and os.path.isfile(dockerfile):
      _ensure_ar_login(mirror)
      with open(dockerfile) as f:
        original = f.read()
      with open(dockerfile, "w") as f:
        f.write(_rewrite_dockerfile_for_mirror(original, mirror))
    return _run(
        ["docker", "buildx", "build", "--network=host", "--load", "-t", tag, ctx],
        timeout=timeout,
    )
  finally:
    shutil.rmtree(staging, ignore_errors=True)


def remove_image(tag: str, *, timeout: float = 120.0) -> ExecResult:
  """Removes a built image to bound vfs disk usage.

  dockerd runs with ``--storage-driver=vfs`` (the only driver that comes up nested
  in the iris task), and vfs does NOT share layers between images -- disk ≈ Σ of
  every image kept. An eval that builds all 70 TB-dev task images without freeing
  them needs hundreds of GB; removing each task's image once its samples are graded
  keeps the footprint to ~one image at a time, so the job fits a small disk.
  """
  return _run(["docker", "rmi", "-f", tag], timeout=timeout)


# All TB-dev task images we build share this tag prefix (see eval/tb_tasks.py:
# ``image_tag=f"ota-tb/{task_id}:latest"``). A prefix-based sweep lets a job free
# every image it built regardless of which tasks ran, as a backstop to the
# per-task ``remove_image`` calls.
_OTA_IMAGE_PREFIX = "ota-tb/"


def prune_ota_images(*, prefix: str = _OTA_IMAGE_PREFIX, timeout: float = 300.0) -> int:
  """Force-removes every locally-built TB task image (``<prefix>*``).

  A belt-and-suspenders cleanup for the end of a run (or a ``finally``): the
  per-task ``remove_image`` already bounds disk during the loop, but a crash
  between build and removal -- or the RL path, which builds all task images up
  front and reuses them across steps -- can leave images resident. dockerd's vfs
  store is per-container and dies with the ephemeral iris task, so this only
  matters to bound disk *within* a run; it never persists across jobs.

  Returns the number of image tags removed. Best-effort: never raises (a cleanup
  failure must not mask the real result), and a no-op if dockerd isn't up.
  """
  if shutil.which("docker") is None:
    return 0
  listed = _run(
      ["docker", "images", "--filter", f"reference={prefix}*", "-q", "--no-trunc"],
      timeout=30,
  )
  if listed.exit_code != 0:
    return 0
  ids = sorted({i for i in listed.stdout.split() if i})
  if not ids:
    return 0
  _run(["docker", "rmi", "-f", *ids], timeout=timeout)
  return len(ids)


class LocalUnsafeSandbox:
  """Plain subprocess execution with NO isolation. Dev/testing only.

  Runs commands directly in a temp dir on the host. Use ONLY to exercise the
  agent-loop / grading logic with trusted commands; never on model-generated code.
  """

  def __init__(self, workdir: str | None = None):
    import tempfile

    self.workdir = workdir or tempfile.mkdtemp(prefix="ota-local-")

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    return _run(["bash", "-lc", command], timeout=timeout, input_text=None)

  def close(self) -> None:
    pass


def make_sandbox(image: str | None = None, **kwargs) -> Sandbox:
  """Returns a sandbox per the ``OTA_SANDBOX`` env (default ``gvisor``).

  Args:
    image: the Docker image for the gvisor sandbox (required for ``gvisor``).
    **kwargs: forwarded to the sandbox constructor.

  Raises:
    ValueError: for an unknown ``OTA_SANDBOX`` or a missing image.
  """
  kind = os.environ.get("OTA_SANDBOX", "gvisor").lower()
  if kind == "gvisor":
    if not image:
      raise ValueError("gvisor sandbox requires an image.")
    return GvisorContainerSandbox(image, **kwargs)
  if kind == "local":
    return LocalUnsafeSandbox(**{k: v for k, v in kwargs.items() if k == "workdir"})
  raise ValueError(f"Unknown OTA_SANDBOX={kind!r} (expected gvisor|local).")
