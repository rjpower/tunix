"""Stage a remote (s3://) orbax checkpoint to local disk so orbax can read it.

tunix loads checkpoints through orbax, whose ``build_kvstore_tspec`` runs
``os.path.normpath`` and mangles ``s3://bucket/x`` -> ``s3:/bucket/x`` (etils.epath
has no s3 backend), so orbax CANNOT read an ``s3://`` root directly. On the CW
GPU cluster there is no gs:// access either -- only R2 (s3://). tensorstore's s3
driver DOES read R2 fine, so we mirror the checkpoint s3->local NVMe in-job and
hand orbax the local dir. (gs:// and already-local paths pass through unchanged.)

R2 (Cloudflare) creds: tensorstore's s3 driver reads AWS_* env; we map R2_*-> AWS_*
+ the marin-na R2 endpoint when AWS_* is unset.
"""

import os
import time

_R2_ENDPOINT = "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"


def _map_r2_to_aws_env():
  if "R2_ACCESS_KEY_ID" in os.environ:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ["R2_ACCESS_KEY_ID"])
  if "R2_SECRET_ACCESS_KEY" in os.environ:
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY", os.environ["R2_SECRET_ACCESS_KEY"]
    )
  os.environ.setdefault("AWS_ENDPOINT_URL", _R2_ENDPOINT)
  os.environ.setdefault("AWS_REGION", "auto")


def _s3_kvstore(bucket: str, prefix: str):
  import tensorstore as ts  # noqa: g-import-not-at-top  pylint: disable=g-import-not-at-top

  _map_r2_to_aws_env()
  return ts.KvStore.open({
      "driver": "s3",
      "bucket": bucket,
      "path": (prefix.rstrip("/") + "/") if prefix else "",
      "endpoint": os.environ.get("AWS_ENDPOINT_URL", _R2_ENDPOINT),
      "aws_region": os.environ.get("AWS_REGION", "auto"),
  }).result()


def stage_checkpoint_if_remote(ckpt_dir, local_root="./_staged_ckpt"):
  """Mirror an ``s3://`` checkpoint root to ``local_root`` and return the local
  path. Non-s3 paths (local, gs://) and ``None`` pass through unchanged.

  Every key under the s3 prefix is written to the matching relative path under
  ``local_root``, so ``CheckpointManager(root_directory=local_root)`` sees the
  same step subdirs orbax expects.
  """
  if not ckpt_dir or not ckpt_dir.startswith("s3://"):
    return ckpt_dir

  bucket, _, prefix = ckpt_dir[len("s3://") :].partition("/")
  kv = _s3_kvstore(bucket, prefix)
  keys = [k.decode() if isinstance(k, bytes) else k
          for k in kv.list().result()]
  if not keys:
    raise RuntimeError(f"No objects under {ckpt_dir!r} to stage.")

  os.makedirs(local_root, exist_ok=True)
  t0 = time.time()
  total_bytes = 0
  for i, key in enumerate(keys):
    data = bytes(kv.read(key).result().value)
    dest = os.path.join(local_root, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
      f.write(data)
    total_bytes += len(data)
    if (i + 1) % 25 == 0 or (i + 1) == len(keys):
      print(f"[stage] {i + 1}/{len(keys)} files, "
            f"{total_bytes / 1e9:.2f}GB in {time.time() - t0:.0f}s", flush=True)

  rate = (total_bytes / 1e9) / max(time.time() - t0, 1e-3)
  print(f"[stage] DONE {ckpt_dir} -> {local_root}: {len(keys)} files, "
        f"{total_bytes / 1e9:.2f}GB ({rate:.2f} GB/s)", flush=True)
  return local_root
