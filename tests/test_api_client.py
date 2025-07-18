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

r"""
Unit tests for API client
"""

import pytest
import json
from pennylane_ionq import api_client
from pennylane_ionq.api_client import (
    requests,
    Job,
    Resource,
    Field,
    ResourceManager,
    ObjectAlreadyCreatedException,
    MethodNotSupportedException,
)

from unittest.mock import MagicMock


status_codes = requests.status_codes.codes


@pytest.fixture
def client():
    return api_client.APIClient(api_key="test")


SAMPLE_JOB_CREATE_RESPONSE = {
    "id": "a6a146d0-d64f-42f4-8b17-ec761fbab7fd",
    "status": "ready",
}

SAMPLE_JOB_RESPONSE = {
    "id": "617a1f8b-59d4-435d-aa33-695433d7155e",
    "type": "simulation",
    "status": "running",
    "request": "1490932820",
    "response": "1490932834",
}


class MockResponse:
    """
    A helper class to generate a mock response based on status code. Mocks
    the `json` and `text` attributes of a requests.Response class.
    """

    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]

    @property
    def text(self):
        return json.dumps(self.json())

    def raise_for_status(self):
        raise requests.exceptions.HTTPError()


class MockPOSTResponse(MockResponse):
    possible_responses = {
        201: SAMPLE_JOB_CREATE_RESPONSE,
        400: {},
        401: {},
        409: {},
        500: {},
    }


class MockGETResponse(MockResponse):
    possible_responses = {
        200: SAMPLE_JOB_RESPONSE,
        401: {},
        404: {},
        500: {},
    }

    status_code = None

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return self.possible_responses[self.status_code]

    def raise_for_status(self):
        raise requests.exceptions.HTTPError()


class TestAPIClient:
    def test_init_default_client(self):
        """
        Test that initializing a default client generates an APIClient with the expected params.
        """
        client = api_client.APIClient(api_key="test")
        assert client.USER_AGENT == "pennylane-ionq-api-client/0.4"
        # TODO: Update to production hostname
        assert client.HOSTNAME == "api-staging.ionq.co/v0.4"
        assert client.BASE_URL.startswith("https://")
        assert client.BASE_URL.endswith(client.HOSTNAME)
        assert client.HEADERS["User-Agent"] == client.USER_AGENT
        assert client.TIMEOUT_SECONDS == 600

    def test_set_authorization_header(self):
        """
        Test that the authentication token is added to the header correctly.
        """
        client = api_client.APIClient(api_key="test")

        authentication_token = MagicMock()
        client.set_authorization_header(authentication_token)
        assert client.HEADERS["Authorization"] == "apiKey {}".format(authentication_token)

    def test_join_path(self, client):
        """
        Test that two paths can be joined and separated by a forward slash.
        """
        assert client.join_path("jobs") == "{client.BASE_URL}/jobs".format(client=client)


class TestResourceManager:
    def test_init(self):
        """
        Test that a resource manager instance can be initialized correctly with a resource and
        client instance. Assets that both manager.resource and manager.client are set.
        """
        resource = MagicMock()
        client = MagicMock()
        manager = ResourceManager(resource, client)

        assert manager.resource == resource
        assert manager.client == client

    def test_join_path(self):
        """
        Test that the resource path can be joined corectly with the base path.
        """
        mock_resource = MagicMock()
        mock_resource.PATH = "some-path"

        manager = ResourceManager(mock_resource, MagicMock())
        assert manager.join_path("test") == "some-path/test"

    def test_get_unsupported(self):
        """
        Test a GET request with a resource that does not support it. Asserts that
        MethodNotSupportedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.get(1)

    @pytest.mark.parametrize("resource_id", [1, None])
    @pytest.mark.parametrize("params", [{}, {"sharpen": True}, {"sharpen": False}])
    def test_get(self, monkeypatch, resource_id, params):
        """
        Test a successful GET request. Tests that manager.handle_response is being called with
        the correct Response object.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.get = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ("GET",)

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.get(resource_id=resource_id, params=params)

        # TODO test that this is called with correct path
        mock_client.get.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response, params)

    def test_create_unsupported(self):
        """
        Test a POST (create) request with a resource that does not support that type or request.
        Asserts that MethodNotSupportedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(MethodNotSupportedException):
            manager.create()

    def test_create_id_already_exists(self):
        """
        Tests that once an object is created, create method can not be called again. Asserts that
        ObjectAlreadyCreatedException is raised.
        """
        mock_resource = MagicMock()
        mock_resource.SUPPORTED_METHODS = ("POST",)
        mock_resource.id = MagicMock()
        manager = ResourceManager(mock_resource, MagicMock())
        with pytest.raises(ObjectAlreadyCreatedException):
            manager.create()

    def test_create(self, monkeypatch):
        """
        Tests a successful POST (create) method. Asserts that handle_response is called with the
        correct Response object.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)

        mock_resource.SUPPORTED_METHODS = ("POST",)
        mock_resource.id = None

        manager = ResourceManager(mock_resource, mock_client)
        monkeypatch.setattr(manager, "handle_response", MagicMock())

        manager.create()

        # TODO test that this is called with correct path and params
        mock_client.post.assert_called_once()
        manager.handle_response.assert_called_once_with(mock_response)

    def test_handle_response(self, monkeypatch):
        """
        Tests that a successful response initiates a call to handle_success_response, and that an
        error response initiates a call to handle_error_response.
        """
        mock_resource = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_handle_success_response = MagicMock()
        mock_handle_error_response = MagicMock()

        manager = ResourceManager(mock_resource, mock_client)

        monkeypatch.setattr(manager, "handle_success_response", mock_handle_success_response)

        monkeypatch.setattr(manager, "handle_error_response", mock_handle_error_response)

        manager.handle_response(mock_response)
        assert manager.http_response_data == mock_response.json()
        assert manager.http_response_status_code == mock_response.status_code
        mock_handle_error_response.assert_called_once_with(mock_response)

        mock_response.status_code = 200
        manager.handle_response(mock_response)
        mock_handle_success_response.assert_called_once_with(mock_response, params=None)

    def test_handle_refresh_data(self):
        """
        Tests the ResourceManager.refresh_data method.
        """
        # start by setting up mocks
        mock_response = MagicMock()
        mock_response.json.return_value = {"some": "result"}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        fields = [MagicMock(name=f"field_{i}") for i in range(5)]
        for i, field in enumerate(fields):
            field.name = f"key_{i}"

        # mock data has matching keys to fields
        mock_data = {f"key_{i}": f"value_{i}" for i in range(5)}

        mock_url = "probability_url"
        mock_data["results"] = {"probabilities": {"url": mock_url}}

        mock_resource = MagicMock()
        mock_resource.refresh_data = MagicMock()
        mock_resource.fields = fields

        # instantiate resource manager
        manager = ResourceManager(mock_resource, mock_client)
        manager.join_path = MagicMock(return_value="joined_url")

        # call the method under test
        manager.refresh_data(mock_data, params={"foo": "bar"})

        # assert fields set method is called correctly
        for i, field in enumerate(mock_resource.fields):
            if i == 4:
                # last field, expect two calls
                assert field.set.call_count == 2
                field.set.assert_any_call("value_4")
                field.set.assert_any_call({"some": "result"})
            else:
                field.set.assert_called_once_with(mock_data.get(field.name))

        # assert client.get was called correctly
        manager.join_path.assert_called_once_with(mock_url)
        mock_client.get.assert_called_once_with("joined_url", params={"foo": "bar"})

        # assert resource.refresh_data was called
        mock_resource.refresh_data.assert_called_once()

    def test_debug_mode(self, monkeypatch):
        """
        Tests that the client object keeps track of responses and errors when debug mode is enabled.
        """

        class MockException(Exception):
            """
            A mock exception to ensure that the exception raised is the expected one.
            """

            pass

        def mock_raise(exception):
            raise exception

        mock_get_response = MockGETResponse(200)

        monkeypatch.setattr(
            requests,
            "get",
            lambda url, params=None, timeout=None, headers=None: mock_get_response,
        )
        monkeypatch.setattr(
            requests,
            "post",
            lambda url, data=None, timeout=None, headers=None: mock_raise(MockException),
        )

        client = api_client.APIClient(debug=True, api_key="test")

        assert client.DEBUG is True
        assert client.errors == []
        assert client.responses == []

        client.get("")
        assert len(client.responses) == 1
        assert client.responses[0] == mock_get_response

        with pytest.raises(MockException):
            client.post("", {})

        assert len(client.errors) == 1


class TestResource:

    def test_resource_reloaading(self, monkeypatch):
        """Test that ID must be set on resource types when reloading."""

        class NoID(Resource):
            """Dummy API resource without ID set."""

            def __init__(self, client=None, api_key=None):
                """Dummy init."""
                self.fields = (Field("foo", str),)
                super().__init__(client=client, api_key=api_key)

        class WithID(Resource):
            """Dummy API resource without ID set."""

            def __init__(self, client=None, api_key=None):
                """Dummy init."""
                self.fields = (Field("foo", str), Field("id", str))
                super().__init__(client=client, api_key=api_key)

        monkeypatch.setattr(
            requests, "post", lambda url, timeout, headers, data: MockPOSTResponse(201)
        )

        res = NoID(api_key="test")

        with pytest.raises(TypeError, match="Resource does not have an ID"):
            res.reload()

        res = WithID(api_key="test")
        res.reload()

    def test_create_created(self, monkeypatch):
        """
        Tests a successful Job creatioin with a mock POST response. Asserts that all fields on
        the Job instance have been set correctly and match the mock data.
        """
        monkeypatch.setattr(
            requests, "post", lambda url, timeout, headers, data: MockPOSTResponse(201)
        )
        job = Job(api_key="test")
        job.manager.create(params={})
        assert not job.is_complete
        assert not job.is_failed

        keys_to_check = SAMPLE_JOB_CREATE_RESPONSE.keys()
        for key in keys_to_check:
            assert getattr(job, key).value == SAMPLE_JOB_CREATE_RESPONSE[key]

    def test_create_bad_request(self, monkeypatch):
        """
        Tests that the correct error code is returned when a bad request is sent to the server.
        """
        monkeypatch.setattr(
            requests, "post", lambda url, timeout, headers, data: MockPOSTResponse(400)
        )
        job = Job(api_key="test")

        with pytest.raises(Exception):
            job.manager.create(params={})
        assert len(job.manager.errors) == 1
        assert job.manager.errors[0]["status_code"] == 400
        assert job.manager.errors[0]["content"] == MockPOSTResponse(400).json()
