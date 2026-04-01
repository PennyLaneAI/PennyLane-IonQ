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
"""Tests for configurable timeout in APIClient"""
from unittest.mock import Mock, patch
import os

import pytest

from pennylane_ionq.api_client import APIClient, Job


class TestTimeoutConfig:
    """Test timeout configuration in APIClient."""

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_default_timeout(self):
        """Test default timeout is 600 seconds (10 minutes)."""
        client = APIClient()
        assert client.TIMEOUT_SECONDS == 600
        assert client.DEFAULT_TIMEOUT == 600

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_custom_timeout(self):
        """Test custom timeout can be set."""
        client = APIClient(timeout=1800)
        assert client.TIMEOUT_SECONDS == 1800

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_none_timeout_uses_default(self):
        """Test that passing None uses the default timeout."""
        client = APIClient(timeout=None)
        assert client.TIMEOUT_SECONDS == 600

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    def test_zero_timeout(self, mock_get):
        """Test that timeout=0 means no timeout (passed as None to requests)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = APIClient(timeout=0)
        assert client.TIMEOUT_SECONDS == 0

        client.get("test/path")
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] is None

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_negative_timeout_rejected(self):
        """Test that a negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            APIClient(timeout=-1)


class TestTimeoutApplied:
    """Test that timeout is applied to requests."""

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    def test_timeout_passed_to_requests(self, mock_get):
        """Test that the configured timeout is passed to the HTTP call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = APIClient(timeout=120)
        client.get("test/path")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 120


class TestJobPropagation:
    """Test that config is propagated through Job/ResourceManager."""

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_job_passes_config_to_client(self):
        """Test that Job passes timeout and retry config to APIClient."""
        job = Job(timeout=1200, max_retries=5, retry_delay=2.0)
        assert job.manager.client.TIMEOUT_SECONDS == 1200
        assert job.manager.client.max_retries == 5
        assert job.manager.client.retry_delay == 2.0

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_job_default_config(self):
        """Test that Job uses default config when none provided."""
        job = Job()
        assert job.manager.client.TIMEOUT_SECONDS == 600
        assert job.manager.client.max_retries == 3
        assert job.manager.client.retry_delay == 0.5
