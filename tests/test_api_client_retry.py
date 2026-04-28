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
"""Tests for HTTP retry logic in APIClient"""
from unittest.mock import Mock, patch
import os

import pytest
import requests

from pennylane_ionq.api_client import APIClient


class TestAPIClientRetryConfig:
    """Test retry configuration in APIClient."""

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_default_retry_config(self):
        """Test default retry configuration."""
        client = APIClient()
        assert client.max_retries == 3
        assert client.retry_delay == 0.5

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        client = APIClient(max_retries=5, retry_delay=1.0)
        assert client.max_retries == 5
        assert client.retry_delay == 1.0

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_negative_max_retries_rejected(self):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            APIClient(max_retries=-1)

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    def test_negative_retry_delay_rejected(self):
        """Test that negative retry_delay raises ValueError."""
        with pytest.raises(ValueError, match="retry_delay must be non-negative"):
            APIClient(retry_delay=-1.0)


class TestRetryLogic:
    """Test retry behaviour in request method."""

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_retry_on_retriable_status(self, mock_sleep, mock_get):
        """Test that retriable status codes are retried."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [mock_response_503, mock_response_503, mock_response_200]

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 200
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_retry_409_on_get(self, mock_sleep, mock_get):
        """Test that 409 is retried on GET requests."""
        mock_response_409 = Mock()
        mock_response_409.status_code = 409

        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [mock_response_409, mock_response_200]

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 200
        assert mock_get.call_count == 2

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.post")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_no_retry_409_on_post(self, mock_sleep, mock_post):
        """Test that 409 is NOT retried on POST requests."""
        mock_response_409 = Mock()
        mock_response_409.status_code = 409

        mock_post.return_value = mock_response_409

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.post("test/path", {"data": "value"})

        assert response.status_code == 409
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.post")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_no_retry_503_on_post(self, mock_sleep, mock_post):
        """Test that 503 is NOT retried on POST requests (non-idempotent)."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        mock_post.return_value = mock_response_503

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.post("test/path", {"data": "value"})

        assert response.status_code == 503
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_no_retry_on_400(self, mock_sleep, mock_get):
        """Test that non-retriable client errors are not retried."""
        mock_response_400 = Mock()
        mock_response_400.status_code = 400

        mock_get.return_value = mock_response_400

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 400
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_retry_on_500(self, mock_sleep, mock_get):
        """Test that 500 Internal Server Error is retried."""
        mock_response_500 = Mock()
        mock_response_500.status_code = 500

        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [mock_response_500, mock_response_200]

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 200
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_exponential_backoff(self, mock_sleep, mock_get):
        """Test that exponential backoff is applied between retries."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [mock_response_503, mock_response_503, mock_response_503, mock_response_200]

        client = APIClient(max_retries=3, retry_delay=1.0)
        response = client.get("test/path")

        calls = mock_sleep.call_args_list
        assert len(calls) == 3
        assert calls[0][0][0] == 1.0   # delay * 2^0
        assert calls[1][0][0] == 2.0   # delay * 2^1
        assert calls[2][0][0] == 4.0   # delay * 2^2

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_exhausted_retries_returns_last_response(self, mock_sleep, mock_get):
        """Test that exhausted retries return the last response."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        mock_get.return_value = mock_response_503

        client = APIClient(max_retries=2, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 503
        assert mock_get.call_count == 3  # initial + 2 retries

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep, mock_get):
        """Test that connection errors are retried."""
        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response_200,
        ]

        client = APIClient(max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 200
        assert mock_get.call_count == 2

    @patch.dict(os.environ, {"IONQ_API_KEY": "test_key"})
    @patch("pennylane_ionq.api_client.requests.get")
    @patch("pennylane_ionq.api_client.time.sleep")
    def test_retry_logs_errors_in_debug_mode(self, mock_sleep, mock_get):
        """Test that retriable status codes are logged to errors when debug mode is enabled."""
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        mock_response_200 = Mock()
        mock_response_200.status_code = 200

        mock_get.side_effect = [mock_response_503, mock_response_200]

        client = APIClient(debug=True, max_retries=3, retry_delay=0.1)
        response = client.get("test/path")

        assert response.status_code == 200
        assert len(client.errors) == 1
        assert "Retriable status 503" in client.errors[0][2]
