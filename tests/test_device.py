# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that plugin devices are accessible and integrate with PennyLane"""
import json
import numpy as np
import pennylane as qml
import pytest
import requests

from conftest import shortnames
from pennylane_ionq.api_client import JobExecutionError, ResourceManager, Job, Field
from pennylane_ionq.device import QPUDevice, IonQDevice

FAKE_API_KEY = "ABC123"


class TestDevice:
    """Tests for the IonQDevice class."""

    @pytest.mark.parametrize(
        "wires,histogram",
        [
            (1, {"0": 0.6, "1": 0.4}),
            (2, {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}),
            (4, {"0": 0.413, "6": 0.111, "15": 0.476}),
            (4, {"7": 0.413, "3": 0.111, "2": 0.476}),
        ],
    )
    def test_generate_samples_qpu_device(self, wires, histogram):
        """Test that the generate_samples method for QPUDevices shuffles the samples between calls."""

        dev = QPUDevice(wires, shots=1024, api_key=FAKE_API_KEY)
        dev.histogram = histogram

        sample1 = dev.generate_samples()
        assert dev.histogram == histogram # make sure histogram is still the same
        sample2 = dev.generate_samples()
        assert not np.all(sample1 == sample2)  # some rows are different

        unique_outcomes1 = np.unique(sample1, axis=0)
        unique_outcomes2 = np.unique(sample2, axis=0)
        assert np.all(unique_outcomes1 == unique_outcomes2)  # possible outcomes are the same

        sorted_outcomes1 = np.sort(sample1, axis=0)
        sorted_outcomes2 = np.sort(sample2, axis=0)
        assert np.all(sorted_outcomes1 == sorted_outcomes2)  # set of outcomes is the same


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortnames)
    def test_load_device(self, d):
        """Test that the device loads correctly"""
        dev = qml.device(d, wires=2, shots=1024)
        assert dev.num_wires == 2
        assert dev.shots == 1024
        assert dev.short_name == d

    @pytest.mark.parametrize("d", shortnames)
    def test_args(self, d):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device(d)

        # IonQ devices do not allow shots=None
        with pytest.raises(ValueError, match="does not support analytic"):
            qml.device(d, wires=1, shots=None)

    def test_emptycircuit_warning(self, mocker):
        """Test warning raised on submission of an empty circuit."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,))

        with pytest.warns(UserWarning, match=r"Circuit is empty."):
            dev.apply([])

    @pytest.mark.parametrize("shots", [100, 500, 8192])
    def test_shots(self, shots, monkeypatch, mocker, tol):
        """Test that shots are correctly specified when submitting a job to the API."""

        monkeypatch.setattr(
            requests, "post", lambda url, timeout, data, headers: (url, data, headers)
        )
        monkeypatch.setattr(ResourceManager, "handle_response", lambda self, response: None)
        monkeypatch.setattr(Job, "is_complete", True)

        def fake_response(self, resource_id=None):
            """Return fake response data"""
            fake_json = {"histogram": {"0": 1}}
            setattr(self.resource, "data", type("data", tuple(), {"value": fake_json})())

        monkeypatch.setattr(ResourceManager, "get", fake_response)

        dev = qml.device("ionq.simulator", wires=1, shots=shots, api_key="test")

        @qml.qnode(dev)
        def circuit():
            """Reference QNode"""
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(requests, "post")
        circuit()
        assert json.loads(spy.call_args[1]["data"])["shots"] == shots

    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_circuit(self, shots, requires_api, tol):
        """Test that devices provide correct result for a simple circuit"""
        dev = qml.device("ionq.simulator", wires=1, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_ordering(self, shots, requires_api, tol):
        """Test that probabilities are returned with the correct qubit ordering"""
        dev = qml.device("ionq.simulator", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=1)
            return qml.probs(wires=[0, 1])

        res = circuit()
        assert np.allclose(res, np.array([0.0, 1.0, 0.0, 0.0]), **tol)

    @pytest.mark.parametrize("d", shortnames)
    def test_prob_no_results(self, d):
        """Test that the prob attribute is
        None if no job has yet been run."""
        dev = qml.device(d, wires=1, shots=1)
        assert dev.prob is None
