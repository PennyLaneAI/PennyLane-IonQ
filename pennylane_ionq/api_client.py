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

"""
APIClient library
=================

**Module name:** :mod:`pennylane_ionq.api_client`

.. currentmodule:: pennylane_ionq.api_client


This module provides a thin client that communicates with the IonQ Platform API over the HTTP
protocol, based on the requests module. It also provides helper classes to facilitate interacting
with this API via the Resource subclasses, as well as the ResourceManager wrapper around APIClient
that is available for each resource.

A single :class:`~.APIClient` instance can be used throughout one's session in the application.
The application will attempt to configure the :class:`~.APIClient` instance using a configuration
file or defaults, but the user can choose to override various parameters of the :class:`~.APIClient`
manually.

Classes
-------

.. autosummary::
   APIClient
   Resource
   ResourceManager
   Field
   Job
   JobResult
   JobCircuit

Exceptions
----------

.. autosummary::
   MethodNotSupportedException
   ObjectAlreadyCreatedException
   JobNotQueuedError
   JobExecutionError

----
"""

import urllib
import json
import warnings
import os
import time

import dateutil.parser

import requests

# HTTP status codes that are retriable, based on qiskit-ionq implementation.
# Includes Cloudflare-specific codes since IonQ uses Cloudflare.
RETRIABLE_STATUS_CODES = (
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    *range(520, 530),  # Cloudflare-specific errors
)

# 409 Conflict is retriable only for GET requests (Cloudflare DNS resolution errors).
RETRIABLE_FOR_GET = (409,)


def join_path(base_path, path):
    """
    Joins two paths, a base path and another path and returns a string.

    Args:
        base_path (str): The left side of the joined path.
        path (str): The right side of the joined path.

    Returns:
        str: A joined path.
    """
    return urllib.parse.urljoin(f"{base_path}/", path)


class MethodNotSupportedException(TypeError):
    """
    Raised when a ResourceManager method is not supported for a
    particular Resource.
    """


class ObjectAlreadyCreatedException(TypeError):
    """
    Raised when an object has already been created but the user
    is attempting to create it again.
    """


class JobNotQueuedError(Exception):
    """
    Raised when a job is not successfully queued for whatever reason.
    """


class JobExecutionError(Exception):
    """
    Raised when job execution failed and a job result does not exist.
    """


class APIClient:  # pylint: disable=too-many-instance-attributes
    """
    Allows the user to connect to the IonQ Platform API.

    Keyword Args:
        api_key (str): IonQ cloud platform API key
        timeout (float): Request timeout in seconds (default: 600, i.e., 10 minutes).
            A value of 0 means no timeout.
        max_retries (int): Maximum number of retries for retriable HTTP errors (default: 3)
        retry_delay (float): Base delay in seconds between retries (default: 0.5)
    """

    USER_AGENT = "pennylane-ionq-api-client/0.4"
    HOSTNAME = "api.ionq.co/v0.4"
    BASE_URL = f"https://{HOSTNAME}"
    DEFAULT_TIMEOUT = 600

    def __init__(self, **kwargs):
        self.AUTHENTICATION_TOKEN = (
            kwargs.get("api_key", None)
            or os.getenv("PENNYLANE_IONQ_API_KEY")
            or os.getenv("IONQ_API_KEY")
        )
        self.DEBUG = False

        if "IONQ_DEBUG" in os.environ:
            # if provided, get debug mode from environment variable
            self.DEBUG = json.loads(os.getenv("IONQ_DEBUG").lower())

        # keyword argument overwrites IONQ_DEBUG environment variable
        self.DEBUG = kwargs.get("debug", self.DEBUG)

        self.HEADERS = {"User-Agent": self.USER_AGENT}

        # Configurable timeout on requests.
        timeout = kwargs.get("timeout")
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        self.TIMEOUT_SECONDS = self.DEFAULT_TIMEOUT if timeout is None else timeout

        # Retry configuration.
        max_retries = kwargs.get("max_retries")
        retry_delay = kwargs.get("retry_delay")
        if max_retries is not None and max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if retry_delay is not None and retry_delay < 0:
            raise ValueError(f"retry_delay must be non-negative, got {retry_delay}")
        self.max_retries = 3 if max_retries is None else max_retries
        self.retry_delay = 0.5 if retry_delay is None else retry_delay

        if self.AUTHENTICATION_TOKEN:
            self.set_authorization_header(self.AUTHENTICATION_TOKEN)
        else:
            raise PermissionError("API key must be provided")

        if self.DEBUG:
            self.errors = []
            self.responses = []

    def set_authorization_header(self, authentication_token):
        """
        Adds the authorization header to the headers dictionary to be included
        with all API requests.

        Args:
            authentication_token (str): an authentication token used to access the API
        """
        self.HEADERS["Authorization"] = f"apiKey {authentication_token}"

    def join_path(self, path):
        """
        Joins a base url with an additional path (e.g., a resource name and ID).

        Args:
            path (str): A path to be joined with ``BASE_URL``

        Returns:
            str: resulting joined path
        """
        return join_path(self.BASE_URL, path)

    def request(self, method, **params):
        """
        Calls ``method`` with ``params`` after applying headers. Records the request type and
        parameters to ``self.errors`` if the request is not successful, and the response to
        ``self.responses`` if a response is returned from the server.

        Implements retry logic with exponential backoff. Status code retries
        apply only to GET requests (see ``RETRIABLE_STATUS_CODES`` and
        ``RETRIABLE_FOR_GET``). Connection error retries apply to all methods.

        Args:
            method: one of ``requests.get`` or ``requests.post``
            **params: the parameters to pass on to the method (e.g. ``url``, ``data``, etc.)

        Returns:
            requests.Response: a response object, or None if no response could be fetched
        """
        supported_methods = (requests.get, requests.post)
        if method not in supported_methods:
            raise TypeError("Unexpected or unsupported method provided")

        params["headers"] = self.HEADERS

        # A timeout of 0 means no timeout; requests expects None for this.
        params["timeout"] = self.TIMEOUT_SECONDS or None

        for attempt in range(self.max_retries + 1):
            try:
                response = method(**params)

                # Only retry on status codes for GET requests; POST is not
                # idempotent, so retrying could create duplicate jobs.
                is_retriable = method == requests.get and (
                    response.status_code in RETRIABLE_STATUS_CODES + RETRIABLE_FOR_GET
                )

                if is_retriable and attempt < self.max_retries:
                    # Exponential backoff.
                    delay = self.retry_delay * (2**attempt)
                    if self.DEBUG:
                        self.errors.append(
                            (
                                method,
                                params,
                                f"Retriable status {response.status_code}, retrying in {delay}s",
                            )
                        )
                    time.sleep(delay)
                    continue

                if self.DEBUG:
                    self.responses.append(response)

                return response

            except Exception as e:
                if self.DEBUG:
                    self.errors.append((method, params, e))

                # Retry only on requests-related exceptions.
                if (
                    isinstance(e, requests.exceptions.RequestException)
                    and attempt < self.max_retries
                ):
                    # Exponential backoff for connection errors.
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise

    def get(self, path, params=None):
        """
        Sends a GET request to the provided path. Returns a response object.

        Args:
            path (str): path to send the GET request to
            params (dict): parameters to include in the request

        Returns:
            requests.Response: A response object, or None if no response could be fetched
        """
        return self.request(requests.get, url=self.join_path(path), params=params)

    def post(self, path, payload):
        """
        Converts payload to a JSON string. Sends a POST request to the provided
        path. Returns a response object.

        Args:
            path (str): path to send the GET request to
            payload: JSON serializable object to be sent to the server

        Returns:
            requests.Response: A response object, or None if no response could be fetched
        """
        return self.request(requests.post, url=self.join_path(path), data=json.dumps(payload))


class ResourceManager:
    """
    Handles all interactions with APIClient by the Resource.
    """

    http_response_data = None
    http_response_status_code = None
    errors = None

    def __init__(self, resource, client=None, api_key=None, **kwargs):
        """
        Initialize the manager with resource and client instances. A client
        instance is used as a persistent HTTP communications object, and a
        resource instance corresponds to a particular type of resource (e.g.,
        Job)

        Args:
            resource: The resource instance to manage.
            client: An optional APIClient instance.
            api_key: API key for authentication.
            **kwargs: Additional arguments passed to APIClient (timeout, max_retries, retry_delay).
        """
        self.resource = resource
        self.client = client or APIClient(api_key=api_key, **kwargs)
        self.errors = []

    def join_path(self, path):
        """
        Joins a resource base path with an additional path (e.g., an ID)
        """
        return join_path(self.resource.PATH, path)

    def get(self, resource_id=None, params=None):
        """
        Attempts to retrieve a particular record by sending a GET
        request to the appropriate endpoint. If successful, the resource
        object is populated with the data in the response.

        Args:
            resource_id (int): the ID of an object to be retrieved
        """
        if "GET" not in self.resource.SUPPORTED_METHODS:
            raise MethodNotSupportedException("GET method on this resource is not supported")

        if resource_id is not None:
            response = self.client.get(self.join_path(str(resource_id)), params=params)
        else:
            response = self.client.get(self.resource.PATH, params=params)

        # we need params later, unfortuantely
        self.handle_response(response, params)

    def create(self, **params):
        """
        Attempts to create a new instance of a resource by sending a POST
        request to the appropriate endpoint.

        Args:
            **params: arbitrary parameters to be passed on to the POST request
        """
        if "POST" not in self.resource.SUPPORTED_METHODS:
            raise MethodNotSupportedException("POST method on this resource is not supported")

        if self.resource.id:
            raise ObjectAlreadyCreatedException("ID must be None when calling create")

        response = self.client.post(self.resource.PATH, params)

        self.handle_response(response)

    def handle_response(self, response, params=None):
        """
        Store the status code on the manager object and handle the response
        based on the status code.

        Args:
            response (requests.Response): a response object to be parsed
        """
        if hasattr(response, "status_code"):
            self.http_response_status_code = response.status_code

            if response.status_code in (200, 201):
                self.http_response_data = response.json()
                self.handle_success_response(response, params=params)
            else:
                try:
                    self.http_response_data = response.json()
                except (json.JSONDecodeError, ValueError):
                    self.http_response_data = None
                self.handle_error_response(response)
        else:
            self.handle_no_response()

    def handle_no_response(self):
        """
        Placeholder method to handle an unsuccessful request (e.g. due to no network connection).
        """
        warnings.warn("Your request could not be completed")

    def handle_success_response(self, response, params=None):
        """
        Handles a successful response by refreshing the instance fields.

        Args:
            response (requests.Response): a response object to be parsed
        """
        self.refresh_data(response.json(), params=params)

    def handle_error_response(self, response):
        """
        Handles an error response that is returned by the server.

        Args:
            response (requests.Response): a response object to be parsed
        """

        try:
            content = response.json()
        except (json.JSONDecodeError, ValueError):
            content = response.text
        error = {"status_code": response.status_code, "content": content}
        self.errors.append(error)
        try:
            response.raise_for_status()
        except Exception as e:
            raise Exception(response.text) from e

    def refresh_data(self, data, params=None):
        """
        Refreshes the instance's attributes with the provided data and
        converts it to the correct type.

        Args:
            data (dict): A dictionary containing keys and values of data to be stored on the object.
        """
        for field in self.resource.fields:
            field.set(data.get(field.name, None))

        results = data.get("results") or {}
        probabilities = results.get("probabilities") or {}
        url = probabilities.get("url")
        if isinstance(url, str) and url:
            resp = self.client.get(self.join_path(url), params=params)
            self.resource.fields[-1].set(resp.json())

        if hasattr(self.resource, "refresh_data"):
            self.resource.refresh_data()


class Resource:
    """
    Base class for an API resource. Should be extended for each resource endpoint.
    """

    SUPPORTED_METHODS = ()
    PATH = ""
    fields = ()

    def __init__(self, client=None, api_key=None, **kwargs):
        """
        Initialize the Resource by populating attributes based on fields and setting a manager.

        Args:
            client (APIClient): An APIClient instance to use as a client.
            api_key: API key for authentication.
            **kwargs: Additional arguments passed to APIClient (timeout, max_retries, retry_delay).
        """
        self.manager = ResourceManager(self, client=client, api_key=api_key, **kwargs)
        for field in self.fields:
            setattr(self, field.name, field)

    def reload(self):
        """
        A helper method to fetch the latest data from the API.
        """
        if not hasattr(self, "id"):
            raise TypeError("Resource does not have an ID")

        if self.id:
            self.manager.get(self.id.value)
        else:
            warnings.warn("Could not reload resource data", UserWarning)


class Field:
    """
    Classifies and cleans data returned by the API.
    """

    value = None

    def __init__(self, name, clean=str):
        """
        Initialize the Field object with a name and a cleaning function.

        Args:
            name (str): A string representing the name of the field (e.g., "created_at").
            clean: A method that returns a cleaned value of the field, of the correct type.
        """
        self.name = name
        self.clean = clean

    def __repr__(self):
        """
        Return the string representation of the value.
        """
        return f"<{self.name} {self.__class__.__name__}: {str(self.value)}>"

    def __bool__(self):
        """
        Use the value to determine boolean state.
        """
        return self.value is not None

    def set(self, value):
        """
        Set the value of the Field to `value`.

        Args:
            value: The value to be stored on the Field object.
        """
        self.value = value

    @property
    def cleaned_value(self):
        """
        Return the cleaned value of the field (for example, an integer or Date
        object)
        """
        return self.clean(self.value) if self.value is not None else None


class Job(Resource):
    """
    API resource corresponding to jobs.
    """

    SUPPORTED_METHODS = ("GET", "POST")
    PATH = "jobs"

    def __init__(self, client=None, api_key=None, **kwargs):
        """
        Initialize the Job resource with a set of pre-defined fields.

        Args:
            client (APIClient): An APIClient instance to use as a client.
            api_key: API key for authentication.
            **kwargs: Additional arguments passed to APIClient (timeout, max_retries, retry_delay).
        """
        self.fields = (
            Field("id", str),
            Field("status", str),
            Field("request", dateutil.parser.parse),
            Field("response", dateutil.parser.parse),
            # it is important that data remain the final item in
            # this tuple to ensure storing results in the correct entry
            Field("data"),
        )

        self.result = None
        self.circuit = None

        super().__init__(client=client, api_key=api_key, **kwargs)

    @property
    def is_complete(self):
        """
        Returns True if the job status is "COMPLETE". Case insensitive. Returns False otherwise.
        """
        return self.status.value and self.status.value.upper() == "COMPLETED"

    @property
    def is_failed(self):
        """
        Returns True if the job status is "FAILED". Case insensitive. Returns False otherwise.
        """
        return self.status.value and self.status.value.upper() == "FAILED"

    def refresh_data(self):
        """
        Refresh the job fields and attach a JobResult and JobCircuit object to the Job instance.
        """
        if self.result is None:
            self.result = JobResult(self.id.value, client=self.manager.client)

        if self.circuit is None:
            self.circuit = JobCircuit(self.id.value, client=self.manager.client)


class JobResult(Resource):
    """
    API resource corresponding to the job result.
    """

    SUPPORTED_METHODS = ("GET",)

    def __init__(self, job_id, client=None):
        """
        Initialize the JobResult resource with a pre-defined field.

        Args:
            job_id (int): The ID of the Job object corresponding to the JobResult object.
        """
        self.id = job_id
        self.fields = (Field("result", json.loads),)

        super().__init__(client=client)


class JobCircuit(Resource):
    """
    API resource corresponding to the job circuit.
    """

    SUPPORTED_METHODS = ("GET",)

    def __init__(self, job_id, client=None):
        """
        Initialize the JobCircuit resource with a pre-defined field.

        Args:
            job_id (int): The ID of the Job object corresponding to the JobResult object.
        """
        self.id = job_id
        self.fields = (Field("circuit"),)

        super().__init__(client=client)
