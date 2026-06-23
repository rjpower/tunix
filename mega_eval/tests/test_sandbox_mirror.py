# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Docker Hub -> Artifact Registry mirror rewrite.

Pure string logic (no docker/jax), so these run on CPU. They pin the rewrite to
the FROM shapes the real OpenThoughts-TB-dev tasks use (explicit ``docker.io/``
and implicit official bases like ``ubuntu:``/``python:*-slim*``) plus the edge
cases the rewrite must NOT touch (other registries, ``scratch``, stage refs).
"""

from absl.testing import absltest
from absl.testing import parameterized

from mega_eval.eval import sandbox

_MIRROR = "us-docker.pkg.dev/hai-gcp-models/docker-mirror"


class MirrorImageRefTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Official images (single-component name) must gain a `library/` segment.
      ("explicit_hub_official", "docker.io/ubuntu:22.04",
       f"{_MIRROR}/library/ubuntu:22.04"),
      ("implicit_official_ubuntu", "ubuntu:24.04",
       f"{_MIRROR}/library/ubuntu:24.04"),
      ("implicit_official_python", "python:3.13-slim-bookworm",
       f"{_MIRROR}/library/python:3.13-slim-bookworm"),
      ("digest_official", "ubuntu@sha256:abc123",
       f"{_MIRROR}/library/ubuntu@sha256:abc123"),
      # Namespaced (org/name) images keep their namespace, no `library/`.
      ("explicit_hub_org", "docker.io/bitnami/nginx:1.27",
       f"{_MIRROR}/bitnami/nginx:1.27"),
      ("implicit_org", "tianon/true:latest", f"{_MIRROR}/tianon/true:latest"),
  )
  def test_rewrites_docker_hub_refs(self, ref, expected):
    self.assertEqual(sandbox._mirror_image_ref(ref, _MIRROR), expected)

  @parameterized.named_parameters(
      ("scratch", "scratch"),
      ("gcr", "gcr.io/proj/img:1"),
      ("ghcr", "ghcr.io/org/img:tag"),
      ("quay", "quay.io/org/img:tag"),
      ("private_port", "registry.local:5000/img:1"),
  )
  def test_leaves_non_docker_hub_untouched(self, ref):
    self.assertIsNone(sandbox._mirror_image_ref(ref, _MIRROR))

  def test_trailing_slash_in_prefix_is_normalized(self):
    self.assertEqual(
        sandbox._mirror_image_ref("ubuntu:22.04", _MIRROR + "/"),
        f"{_MIRROR}/library/ubuntu:22.04",
    )


class RewriteDockerfileTest(absltest.TestCase):

  def test_rewrites_from_preserving_flags_and_stage(self):
    src = (
        "FROM --platform=linux/amd64 python:3.13-slim-bookworm AS builder\n"
        "RUN pip install uv\n"
        "FROM docker.io/ubuntu:22.04\n"
        "COPY --from=builder /app /app\n"
    )
    out = sandbox._rewrite_dockerfile_for_mirror(src, _MIRROR)
    self.assertIn(
        f"FROM --platform=linux/amd64 {_MIRROR}/library/python:3.13-slim-bookworm AS builder",
        out,
    )
    self.assertIn(f"FROM {_MIRROR}/library/ubuntu:22.04", out)
    # Non-FROM lines, incl. the intra-build stage ref, are untouched.
    self.assertIn("RUN pip install uv", out)
    self.assertIn("COPY --from=builder /app /app", out)
    self.assertTrue(out.endswith("\n"))

  def test_case_insensitive_from_keyword(self):
    out = sandbox._rewrite_dockerfile_for_mirror("from ubuntu:22.04\n", _MIRROR)
    self.assertEqual(out, f"from {_MIRROR}/library/ubuntu:22.04\n")

  def test_no_trailing_newline_preserved(self):
    out = sandbox._rewrite_dockerfile_for_mirror("FROM ubuntu:22.04", _MIRROR)
    self.assertEqual(out, f"FROM {_MIRROR}/library/ubuntu:22.04")

  def test_non_from_lines_unchanged(self):
    src = "# a comment\nENV X=1\nRUN echo hi\n"
    self.assertEqual(sandbox._rewrite_dockerfile_for_mirror(src, _MIRROR), src)


if __name__ == "__main__":
  absltest.main()
