# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live gVisor (runsc) validation inside an iris TPU task.

TPU tasks run ``--privileged`` (iris adds it for accelerators), which is what
rootful gVisor + a task-local dockerd need. This smoke bootstraps the SAME
binaries the custom task image (`docker/Dockerfile.agent-task`) bakes in --
Docker's static binaries + runsc -- at runtime, then exercises the real
`eval/sandbox.py` paths end-to-end so we validate the mechanism without first
publishing the custom image to a registry:

  1. install docker + runsc, register the runsc Docker runtime,
  2. start dockerd (eval.sandbox.ensure_dockerd),
  3. run a container under runsc and PROVE gVisor isolation by comparing the
     kernel `uname -r` under runsc (gVisor's emulated kernel) vs runc (the host
     kernel) -- a different kernel means gVisor is interposing on syscalls,
  4. drive a GvisorContainerSandbox (exec + copy_in), the exact class the eval
     harness uses per Terminal-Bench task.

Submit on the smallest TPU slice (privileged, fast to schedule):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 60GB --disk 60GB --max-retries 1 --job-name ota-gvisor-smoke \
      -- python -m eval.gvisor_smoke
"""

import os

from mega_eval.eval.sandbox import (
    GvisorContainerSandbox,
    _run,
    ensure_dockerd,
    ensure_sandbox_runtime,
)


def _log(msg: str) -> None:
  print(f"[gvisor-smoke] {msg}", flush=True)


def main() -> None:
  _log(f"uid={os.getuid()} (privileged TPU task expected to be root)")
  # No-op on the custom task image; installs docker+runsc on the stock iris image.
  ensure_sandbox_runtime()

  v = _run(["runsc", "--version"], timeout=30)
  _log(f"runsc --version -> exit={v.exit_code}\n{v.stdout}{v.stderr}")

  ensure_dockerd()
  _log("dockerd is up")

  # gVisor isolation proof: kernel under runsc vs runc.
  host = _run(["uname", "-r"], timeout=15).stdout.strip()
  _log(f"host kernel (task VM): {host}")
  runc = _run(["docker", "run", "--rm", "alpine", "uname", "-r"], timeout=300)
  _log(f"runc container kernel: {runc.stdout.strip()!r} (exit={runc.exit_code}) {runc.stderr[-200:]}")
  gv = _run(["docker", "run", "--rm", "--runtime=runsc", "alpine", "uname", "-r"], timeout=300)
  _log(f"runsc container kernel: {gv.stdout.strip()!r} (exit={gv.exit_code}) {gv.stderr[-300:]}")

  procver = _run(
      ["docker", "run", "--rm", "--runtime=runsc", "alpine", "sh", "-c", "cat /proc/version; dmesg 2>/dev/null | head -3"],
      timeout=120,
  )
  _log(f"runsc /proc/version + dmesg:\n{procver.stdout}")

  isolated = gv.exit_code == 0 and gv.stdout.strip() and gv.stdout.strip() != runc.stdout.strip()
  _log(f"GVISOR ISOLATION: {'CONFIRMED (runsc kernel != host kernel)' if isolated else 'NOT CONFIRMED'}")

  # Exercise the production sandbox class. Use a bash-bearing image (TB task
  # images ship bash; alpine is busybox-only, which is why GvisorContainerSandbox's
  # `bash -lc` would fail there -- an image limitation, not a sandbox bug).
  sandbox_image = "debian:stable-slim"
  _log(f"exercising GvisorContainerSandbox({sandbox_image})")
  sb = GvisorContainerSandbox(sandbox_image, workdir="/root")
  try:
    r = sb.exec("echo from-sandbox && uname -r && id")
    _log(f"sandbox.exec -> exit={r.exit_code}\n{r.stdout}{r.stderr[-200:]}")
    # copy_in a local file then read it back inside the sandbox.
    with open("/tmp/ota_probe.txt", "w") as f:
      f.write("hello-from-host\n")
    cp = sb.copy_in("/tmp/ota_probe.txt", "/root/probe.txt")
    rb = sb.exec("cat /root/probe.txt")
    _log(f"sandbox.copy_in exit={cp.exit_code}; read-back={rb.stdout.strip()!r}")
  finally:
    sb.close()

  _log("DONE")


if __name__ == "__main__":
  main()
