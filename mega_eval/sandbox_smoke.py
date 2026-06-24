"""M1d infra gate: does the gVisor sandbox actually boot inside a marin TPU pod?

The agentic Terminal-Bench rollout runs the model's shell commands in a per-task
gVisor container (docker run --runtime=runsc) booted *inside* the iris TPU task.
That needs a nested dockerd + runsc to come up under whatever privileges the pod
actually has (SYS_PTRACE always, SYS_RESOURCE on TPU; the iris k8s backend does
NOT set privileged:true). The sandbox code is engineered for this (runsc
--platform=ptrace --ignore-cgroups, dockerd --storage-driver=vfs), but it's never
been confirmed on a marin TPU pod. This minimal smoke proves (or refutes) it
before we build the whole agentic collector on top.

Stages: ensure_sandbox_runtime (download docker+runsc on the stock image) ->
ensure_dockerd (nested) -> boot ubuntu under runsc -> exec a command -> a docker
build (the per-task image path) -> teardown. Any failure prints the dockerd log.
"""

import sys
import traceback

from mega_eval.eval import sandbox


def _stage(msg):
  print(f"[sbx] {msg}", flush=True)


def main():
  try:
    _stage("ensure_sandbox_runtime() (download docker+runsc+buildx, register runtime)")
    sandbox.ensure_sandbox_runtime()
    _stage("ensure_dockerd() (start nested dockerd, wait for ready)")
    sandbox.ensure_dockerd()
    _stage("dockerd READY")

    _stage("boot ubuntu:22.04 under runsc + exec")
    sb = sandbox.GvisorContainerSandbox(image="ubuntu:22.04")
    try:
      r = sb.exec(
          "echo HELLO_FROM_GVISOR; uname -a; id; "
          "cat /proc/version | head -1; dmesg 2>/dev/null | head -1 || true"
      )
      _stage(f"exec exit={r.exit_code}")
      _stage(f"  stdout: {r.stdout.strip()}")
      if r.stderr.strip():
        _stage(f"  stderr: {r.stderr.strip()}")
      if r.exit_code != 0 or "HELLO_FROM_GVISOR" not in r.stdout:
        raise RuntimeError("exec did not return the sentinel")
      # gVisor identifies itself in /proc/version as "gVisor"; log whether we see it.
      _stage(f"  gVisor kernel: {'YES' if 'gVisor' in r.stdout else 'NOT DETECTED (may be runc fallback)'}")
    finally:
      sb.close()
    _stage("container torn down")

    # The per-task path also docker-builds each task image; smoke that too.
    _stage("docker build a trivial image (the per-task build path)")
    import os
    import tempfile
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "Dockerfile"), "w") as f:
      f.write("FROM ubuntu:22.04\nRUN echo built-in-image > /built.txt\n")
    br = sandbox.build_image(d, "ota-sbx-smoke:latest")
    _stage(f"build exit={br.exit_code}")
    if br.exit_code != 0:
      _stage(f"  build stderr tail: {br.stderr[-500:]}")
      raise RuntimeError("docker build failed")
    sb2 = sandbox.GvisorContainerSandbox(image="ota-sbx-smoke:latest")
    try:
      r2 = sb2.exec("cat /built.txt")
      _stage(f"built-image exec exit={r2.exit_code} stdout={r2.stdout.strip()}")
      if r2.exit_code != 0 or "built-in-image" not in r2.stdout:
        raise RuntimeError("built image did not run under runsc")
    finally:
      sb2.close()

    _stage("=== SANDBOX SMOKE OK (gVisor boots + execs + builds on marin TPU pod) ===")
    return 0
  except Exception as e:  # pylint: disable=broad-except
    _stage(f"!!! SANDBOX SMOKE FAILED: {e!r}")
    traceback.print_exc()
    try:
      with open("/tmp/dockerd.out") as f:
        tail = "".join(f.readlines()[-40:])
      _stage(f"dockerd log tail:\n{tail}")
    except OSError:
      pass
    return 1


if __name__ == "__main__":
  sys.exit(main())
