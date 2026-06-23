# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for topology host-key derivation on the non-Pathways path."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.utils import topology


class _FakeDevice:
  """Minimal device double exposing standard-JAX attrs."""

  def __init__(self, process_index=None, slice_index=None, device_id=0):
    if process_index is not None:
      self.process_index = process_index
    if slice_index is not None:
      self.slice_index = slice_index
    self.id = device_id


class DeviceHostKeyNonPathwaysTest(parameterized.TestCase):
  """The non-Pathways branch must derive sane keys from standard JAX attrs."""

  def setUp(self):
    super().setUp()
    # Force the standard-JAX (non-Pathways) branch regardless of environment.
    patcher = mock.patch.object(
        topology, "_is_pathways_backend_used", return_value=False
    )
    self.addCleanup(patcher.stop)
    patcher.start()

  def test_distinct_process_index_yields_distinct_keys(self):
    # Two simulated hosts (process_index 0 and 1), no slice metadata: the keys
    # must differ so host grouping / mesh allocation stays correct off-Pathways.
    key0 = topology._device_host_key(_FakeDevice(process_index=0))
    key1 = topology._device_host_key(_FakeDevice(process_index=1))
    self.assertEqual(key0, (None, 0))
    self.assertEqual(key1, (None, 1))
    self.assertNotEqual(key0, key1)

  def test_slice_index_is_included_when_present(self):
    # On multi-slice TPU (no Pathways) slice_index is exposed and groups hosts
    # by slice.
    key = topology._device_host_key(_FakeDevice(process_index=2, slice_index=3))
    self.assertEqual(key, (3, 2))

  def test_same_slice_different_hosts(self):
    key_a = topology._device_host_key(
        _FakeDevice(process_index=0, slice_index=1)
    )
    key_b = topology._device_host_key(
        _FakeDevice(process_index=1, slice_index=1)
    )
    self.assertEqual(key_a, (1, 0))
    self.assertEqual(key_b, (1, 1))
    self.assertNotEqual(key_a, key_b)

  def test_returns_none_without_process_index(self):
    # A device exposing no process_index (and not Pathways) has no task
    # metadata, so there is no host key.
    self.assertIsNone(topology._device_host_key(_FakeDevice()))


class DeviceHostKeyPathwaysTest(parameterized.TestCase):
  """The Pathways branch parses logical_task from the device repr (preserved)."""

  def test_pathways_branch_parses_logical_task(self):
    class _PathwaysDevice:

      def __init__(self, repr_str, slice_index=None):
        self._repr = repr_str
        if slice_index is not None:
          self.slice_index = slice_index

      def __repr__(self):
        return self._repr

    with mock.patch.object(
        topology, "_is_pathways_backend_used", return_value=True
    ):
      device = _PathwaysDevice(
          "device(0,TPU,...,logical_task=11,slice=3,...)", slice_index=3
      )
      self.assertEqual(topology._device_host_key(device), (3, 11))


if __name__ == "__main__":
  absltest.main()
