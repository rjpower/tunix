# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for sandbox subprocess output handling (_run / _as_text).

Agent-issued commands run arbitrary programs: their stdout/stderr may be non-UTF-8
and they may time out with partial *bytes* output. A strict decode or a str+bytes
concat there crashes the episode/rollout (observed in the eval sweep as
``UnicodeDecodeError`` and ``TypeError: can't concat str to bytes``). These pin the
lenient handling so it never raises.
"""

from absl.testing import absltest

from mega_eval.eval import sandbox


class AsTextTest(absltest.TestCase):

  def test_none_becomes_empty(self):
    self.assertEqual(sandbox._as_text(None), "")

  def test_str_passthrough(self):
    self.assertEqual(sandbox._as_text("hello"), "hello")

  def test_non_utf8_bytes_replaced_not_raised(self):
    # 0xa4 is invalid UTF-8 lead byte — must map to the replacement char, not raise.
    self.assertEqual(sandbox._as_text(b"\xa4done"), "�done")


class RunRobustnessTest(absltest.TestCase):

  def test_non_utf8_command_output_does_not_raise(self):
    res = sandbox._run(["bash", "-lc", r"printf '\xa4ok'"], timeout=10)
    self.assertEqual(res.exit_code, 0)
    self.assertIn("ok", res.stdout)

  def test_timeout_partial_output_does_not_concat_bytes(self):
    # Partial stdout then hang past the timeout: the timeout branch must coerce
    # any bytes before appending "[timeout]" (the str+bytes TypeError regression).
    res = sandbox._run(["bash", "-lc", "printf pre; sleep 5"], timeout=1)
    self.assertTrue(res.timed_out)
    self.assertEqual(res.exit_code, 124)
    self.assertIn("[timeout]", res.stderr)


if __name__ == "__main__":
  absltest.main()
