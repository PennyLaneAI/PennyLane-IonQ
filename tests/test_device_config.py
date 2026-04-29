# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for device configuration (timeout, retry settings)"""
from unittest.mock import Mock, patch

from pennylane_ionq.device import IonQDevice


FAKE_API_KEY = "ABC123"


class TestDeviceConfigInit:
    """Test that IonQDevice accepts configuration parameters."""

    def test_accepts_timeout(self):
        """Test that IonQDevice accepts timeout parameter."""
        dev = IonQDevice(wires=2, shots=100, api_key=FAKE_API_KEY, timeout=1800)
        assert dev.timeout == 1800

    def test_accepts_retry_config(self):
        """Test that IonQDevice accepts retry configuration."""
        dev = IonQDevice(
            wires=2, shots=100, api_key=FAKE_API_KEY,
            max_retries=5, retry_delay=2.0
        )
        assert dev.max_retries == 5
        assert dev.retry_delay == 2.0

    def test_default_config_is_none(self):
        """Test that IonQDevice has None defaults for optional config."""
        dev = IonQDevice(wires=2, shots=100, api_key=FAKE_API_KEY)
        assert dev.timeout is None
        assert dev.max_retries is None
        assert dev.retry_delay is None


class TestDeviceConfigPropagation:
    """Test that config is propagated to Job during job submission."""

    @patch("pennylane_ionq.device.Job")
    def test_submit_job_passes_config(self, mock_job_class):
        """Test that _submit_job passes all config to Job."""
        mock_job = Mock()
        mock_job.is_complete = True
        mock_job.is_failed = False
        mock_job.id.value = "test-id"
        mock_job.data.value = {"0": 1.0}
        mock_job_class.return_value = mock_job

        dev = IonQDevice(
            wires=2, shots=100, api_key=FAKE_API_KEY,
            timeout=1800, max_retries=5, retry_delay=2.0
        )
        dev._submit_job()

        call_kwargs = mock_job_class.call_args[1]
        assert call_kwargs["timeout"] == 1800
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["retry_delay"] == 2.0

    @patch("pennylane_ionq.device.Job")
    def test_submit_job_none_config_uses_defaults(self, mock_job_class):
        """Test that None config values are passed through to APIClient."""
        mock_job = Mock()
        mock_job.is_complete = True
        mock_job.is_failed = False
        mock_job.id.value = "test-id"
        mock_job.data.value = {"0": 1.0}
        mock_job_class.return_value = mock_job

        dev = IonQDevice(wires=2, shots=100, api_key=FAKE_API_KEY)
        dev._submit_job()

        call_kwargs = mock_job_class.call_args[1]
        assert call_kwargs["timeout"] is None
        assert call_kwargs["max_retries"] is None
        assert call_kwargs["retry_delay"] is None
