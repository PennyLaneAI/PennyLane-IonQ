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
import logging
import numpy as np
import pennylane as qml
import pytest
import requests

from conftest import shortnames
from pennylane_ionq.api_client import JobExecutionError, ResourceManager, Job
from pennylane_ionq.device import (
    QPUDevice,
    IonQDevice,
    SimulatorDevice,
    CircuitIndexNotSetException,
)
from pennylane_ionq.ops import GPI, GPI2, MS, XX, YY, ZZ
from pennylane.measurements import SampleMeasurement, ShotCopies
from unittest import mock

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
        dev.histograms = [histogram]

        sample1 = dev.generate_samples()
        assert dev.histograms[0] == histogram  # make sure histogram is still the same
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
        dev = qml.device(d, wires=2)
        assert dev.num_wires == 2
        assert dev.shots.total_shots is None
        assert dev.short_name == d

    @pytest.mark.parametrize("d", shortnames)
    def test_args(self, d):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device(d)

        # IonQ devices allow shots=None
        dev = qml.device(d, wires=1, shots=None)

        # But the execution will raise error
        @qml.qnode(dev)
        def circuit():
            """Reference QNode"""
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="does not support analytic"):
            circuit()

    def test_emptycircuit_warning(self, mocker):
        """Test warning raised on submission of an empty circuit."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,))

        with pytest.warns(UserWarning, match=r"Circuit is empty."):
            dev.apply([])

    def test_failedcircuit(self, monkeypatch):
        monkeypatch.setattr(
            requests, "post", lambda url, timeout, data, headers: (url, data, headers)
        )
        monkeypatch.setattr(ResourceManager, "handle_response", lambda self, response: None)
        monkeypatch.setattr(Job, "is_complete", False)
        monkeypatch.setattr(Job, "is_failed", True)

        dev = IonQDevice(wires=(0,), api_key="test")
        with pytest.raises(JobExecutionError):
            dev._submit_job()

    @pytest.mark.parametrize("shots", [100, 500, 8192])
    def test_shots(self, shots, monkeypatch, mocker, tol):
        """Test that shots are correctly specified when submitting a job to the API."""

        monkeypatch.setattr(
            requests, "post", lambda url, timeout, data, headers: (url, data, headers)
        )
        monkeypatch.setattr(ResourceManager, "handle_response", lambda self, response: None)
        monkeypatch.setattr(Job, "is_complete", True)

        def fake_response(self, resource_id=None, params=None):
            """Return fake response data"""
            fake_json = {"0": 1}
            setattr(self.resource, "data", type("data", tuple(), {"value": fake_json})())

        monkeypatch.setattr(ResourceManager, "get", fake_response)

        dev = qml.device("ionq.simulator", wires=1, api_key="test")

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            """Reference QNode"""
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(requests, "post")
        circuit()
        assert json.loads(spy.call_args[1]["data"])["shots"] == shots

    @pytest.mark.parametrize("error_mitigation", [None, {"debias": True}, {"debias": False}])
    def test_error_mitigation(self, error_mitigation, monkeypatch, mocker):
        """Test that shots are correctly specified when submitting a job to the API."""

        monkeypatch.setattr(
            requests, "post", lambda url, timeout, data, headers: (url, data, headers)
        )
        monkeypatch.setattr(ResourceManager, "handle_response", lambda self, response: None)
        monkeypatch.setattr(Job, "is_complete", True)

        def fake_response(self, resource_id=None, params=None):
            """Return fake response data"""
            fake_json = {"0": 1}
            setattr(self.resource, "data", type("data", tuple(), {"value": fake_json})())

        monkeypatch.setattr(ResourceManager, "get", fake_response)

        dev = qml.device(
            "ionq.qpu",
            wires=1,
            api_key="test",
            error_mitigation=error_mitigation,
        )

        @qml.set_shots(5000)
        @qml.qnode(dev)
        def circuit():
            """Reference QNode"""
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(requests, "post")
        circuit()
        if error_mitigation is not None:
            assert json.loads(spy.call_args[1]["data"])["error_mitigation"] == error_mitigation
        else:
            with pytest.raises(KeyError, match="error_mitigation"):
                json.loads(spy.call_args[1]["data"])["error_mitigation"]

    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_circuit(self, shots, requires_api, tol):
        """Test that devices provide correct result for a simple circuit"""
        dev = qml.device("ionq.simulator", wires=1)

        a = 0.543
        b = 0.123
        c = qml.numpy.array(0.987, requires_grad=False)

        @qml.set_shots(shots)
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
        dev = qml.device("ionq.simulator", wires=2)

        @qml.set_shots(shots)
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
        dev = qml.device(d, wires=1)
        assert dev.prob is None

    @pytest.mark.parametrize(
        "backend",
        [
            "aria-1",
            "aria-2",
            "forte-1",
            "forte-enterprise-1",
            "forte-enterprise-2",
            None,
        ],
    )
    def test_backend_initialization(self, backend):
        """Test that the device initializes with the correct backend."""
        dev = qml.device(
            "ionq.qpu",
            wires=2,
            backend=backend,
        )
        assert dev.backend == backend

    def test_recording_when_pennylane_tracker_active(self, requires_api):
        """Test recording device execution history via pennnylane tracker class."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        dev.tracker = qml.Tracker()
        dev.tracker.active = True
        dev.tracker.reset()
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.probs(wires=[0])
        dev.batch_execute([tape1, tape1])
        assert dev.tracker.history["executions"] == [1, 1]
        assert dev.tracker.history["shots"] == [1024, 1024]
        assert dev.tracker.history["batches"] == [1]
        assert dev.tracker.history["batch_len"] == [2]
        assert len(dev.tracker.history["resources"]) == 2
        assert dev.tracker.history["resources"][0].num_wires == 1
        assert dev.tracker.history["resources"][0].num_gates == 1
        assert dev.tracker.history["resources"][0].depth == 1
        assert dev.tracker.history["resources"][0].gate_types == {"GPI": 1}
        assert dev.tracker.history["resources"][0].gate_sizes == {1: 1}
        assert dev.tracker.history["resources"][0].shots.total_shots == 1024
        assert len(dev.tracker.history["results"]) == 2

    def test_not_recording_when_pennylane_tracker_not_active(self, requires_api):
        """Test recording device not executed when tracker is inactive."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        dev.tracker = qml.Tracker()
        dev.tracker.active = False
        dev.tracker.reset()
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.probs(wires=[0])
        dev.batch_execute([tape1])
        assert dev.tracker.history == {}

    def test_warning_on_empty_circuit(self, requires_api):
        """Test warning are shown when circuit is empty."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            qml.probs(wires=[0])
        with pytest.warns(
            UserWarning,
            match="Circuit is empty. Empty circuits return failures. Submitting anyway.",
        ):
            dev.batch_execute([tape1])

    @mock.patch("logging.Logger.isEnabledFor", return_value=True)
    @mock.patch("logging.Logger.debug")
    def test_batch_execute_logging_when_enabled(
        self,
        mock_logging_debug_method,
        mock_logging_is_enabled_for_method,
        requires_api,
    ):
        """Test logging invoked in batch_execute method."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.probs(wires=[0])
        dev.batch_execute([tape1])
        assert mock_logging_is_enabled_for_method.called
        assert mock_logging_is_enabled_for_method.call_args[0][0] == logging.DEBUG
        mock_logging_debug_method.assert_called()

    def test_batch_execute_probabilities_raises(self, requires_api):
        """Test invoking probability() method raises exception if circuit index not
        previously set when multiple circuits are submitted in one job.
        """
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0.5, wires=[0])
            qml.probs(wires=[0])
        dev.batch_execute([tape1, tape1])
        with pytest.raises(
            CircuitIndexNotSetException,
            match="Because multiple circuits have been submitted in this job, the index of the circuit \
you want to access must be first set via the set_current_circuit_index device method.",
        ):
            dev.probability()

    def test_batch_execute_probabilities(self, requires_api):
        """Test batch_execute method when computing circuit probabilities."""
        dev = SimulatorDevice(wires=(0, 1, 2), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0.5, wires=[0])
            GPI2(0, wires=[1])
            MS(0, 0.5, wires=[1, 2])
            qml.probs(wires=[0, 1, 2])
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            GPI(0.5, wires=[1])
            MS(0, 0.5, wires=[1, 2])
            qml.probs(wires=[0, 1, 2])
        results = dev.batch_execute([tape1, tape2])
        assert np.array_equal(results[0], [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
        assert np.array_equal(results[1], [0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0])
        dev.set_current_circuit_index(0)
        assert np.array_equal(
            dev.probability(),
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
        )
        dev.set_current_circuit_index(1)
        assert np.array_equal(
            dev.probability(),
            [0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0],
        )

    def test_batch_execute_probabilities_with_shot_vector(self, requires_api):
        """Test batch_execute method with shot vector."""
        dev = SimulatorDevice(wires=(0, 1, 2), gateset="native")
        dev._shot_vector = (ShotCopies(1, 3),)
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.probs(wires=[0])
        results = dev.batch_execute([tape1])
        assert len(results[0]) == 3

    def test_batch_execute_variance(self, requires_api):
        """Test batch_execute method when computing variance of an observable."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.var(qml.PauliZ(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            qml.var(qml.PauliZ(0))
        results = dev.batch_execute([tape1, tape2])
        assert results[0] == pytest.approx(0, abs=0.01)
        assert results[1] == pytest.approx(1, abs=0.01)

    def test_batch_execute_expectation_value(self, requires_api):
        """Test batch_execute method when computing expectation value of an observable."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.expval(qml.PauliZ(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            qml.expval(qml.PauliZ(0))
        results = dev.batch_execute([tape1, tape2])
        assert results[0] == pytest.approx(-1, abs=0.1)
        assert results[1] == pytest.approx(0, abs=0.1)

    def test_batch_execute_expectation_value_with_diagonalization_rotations(self, requires_api):
        """Test batch_execute method when computing expectation value of an
        observable that requires rotations for diagonalization."""
        dev = SimulatorDevice(wires=(0,), gateset="qis")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            qml.Hadamard(0)
            qml.expval(qml.PauliX(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            qml.expval(qml.PauliX(0))
        results = dev.batch_execute([tape1, tape2])
        assert results[0] == pytest.approx(1, abs=0.1)
        assert results[1] == pytest.approx(0, abs=0.1)

    def test_batch_execute_invoking_prob_property_raises(self, requires_api):
        """Test invoking prob device property raises exception if circuit index not
        previously set when multiple circuits are submitted in one job.
        """
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.probs(wires=[0])
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            qml.probs(wires=[0])
        dev.batch_execute([tape1, tape2])
        with pytest.raises(
            CircuitIndexNotSetException,
            match="Because multiple circuits have been submitted in this job, the index of the circuit \
you want to access must be first set via the set_current_circuit_index device method.",
        ):
            dev.prob

    def test_batch_execute_prob_property(self, requires_api):
        """Test batch_execute method with invoking invoking prob device property."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.sample(qml.PauliZ(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            qml.sample(qml.PauliZ(0))
        dev.batch_execute([tape1, tape2])
        dev.set_current_circuit_index(0)
        prob0 = dev.prob
        dev.set_current_circuit_index(1)
        prob1 = dev.prob
        np.testing.assert_array_almost_equal(prob0, [0.0, 1.0], decimal=1)
        np.testing.assert_array_almost_equal(prob1, [0.5, 0.5], decimal=1)

    def test_batch_execute_counts(self, requires_api):
        """Test batch_execute method when computing counts."""
        dev = SimulatorDevice(wires=(0,), gateset="native")
        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            qml.counts(qml.PauliZ(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            qml.counts(qml.PauliZ(0))
        results = dev.batch_execute([tape1, tape2])
        assert results[0][-1] == 1024
        assert results[1][-1] == pytest.approx(512, abs=100)

    def test_sample_measurements(self, requires_api):
        """Test branch of code activated by using SampleMeasurement."""

        class CountState(SampleMeasurement):
            def __init__(self, state: str):
                self.state = state  # string identifying the state e.g. "0101"
                wires = list(range(len(state)))
                super().__init__(wires=wires)

            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                counts_mp = qml.counts(wires=self._wires)
                counts = counts_mp.process_samples(samples, wire_order, shot_range, bin_size)
                return float(counts.get(self.state, 0))

            def process_counts(self, counts, wire_order):
                return float(counts.get(self.state, 0))

            def __copy__(self):
                return CountState(state=self.state)

        dev = SimulatorDevice(wires=(0,), gateset="native")

        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0, wires=[0])
            CountState(state="1")

        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0, wires=[0])
            CountState(state="1")

        results = dev.batch_execute([tape1, tape2])
        assert results[0] == 1024
        assert results[1] == pytest.approx(512, abs=100)


class TestJobAttribute:
    """Tests job creation with mocked submission."""

    def test_nonparametrized_tape(self, mocker):
        """Tests job attribute after single paulix tape."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,), target="foo")

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(0)

        dev.apply(tape.operations)

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "qis"
        assert dev.job["target"] == "foo"
        assert dev.job["input"]["qubits"] == 1

        assert len(dev.job["input"]["circuits"][0]) == 1
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "x",
            "target": 0,
        }

    def test_nonparametrized_tape_batch_submit(self, mocker):
        """Tests job attribute after single paulix tape, on batch submit."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,), target="foo")

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(0)

        dev.reset(circuits_array_length=1)
        dev.batch_apply(tape.operations, circuit_index=0)

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "qis"
        assert dev.job["target"] == "foo"
        assert dev.job["input"]["qubits"] == 1

        assert len(dev.job["input"]["circuits"]) == 1
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "x",
            "target": 0,
        }

    def test_parameterized_op(self, mocker):
        """Tests job attribute several parameterized operations."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,))

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.2345, wires=0)
            qml.RY(qml.numpy.array(2.3456), wires=0)

        dev.apply(tape.operations)

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "qis"
        assert dev.job["input"]["qubits"] == 1

        assert len(dev.job["input"]["circuits"][0]["circuit"]) == 2
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "rx",
            "target": 0,
            "rotation": 1.2345,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][1] == {
            "gate": "ry",
            "target": 0,
            "rotation": 2.3456,
        }

    def test_parameterized_op_batch_submit(self, mocker):
        """Tests job attribute several parameterized operations."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0,))

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.2345, wires=0)
            qml.RY(qml.numpy.array(2.3456), wires=0)

        dev.reset(circuits_array_length=1)
        dev.batch_apply(tape.operations, circuit_index=0)

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "qis"
        assert dev.job["input"]["qubits"] == 1

        assert len(dev.job["input"]["circuits"][0]["circuit"]) == 2
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "rx",
            "target": 0,
            "rotation": 1.2345,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][1] == {
            "gate": "ry",
            "target": 0,
            "rotation": 2.3456,
        }

    def test_parameterized_native_op(self, mocker):
        """Tests job attribute several parameterized native operations."""

        def mock_submit_job(*args):
            pass

        mocker.patch("pennylane_ionq.device.IonQDevice._submit_job", mock_submit_job)
        dev = IonQDevice(wires=(0, 1, 2), gateset="native")

        with qml.tape.QuantumTape() as tape:
            GPI(0.1, wires=[0])
            GPI2(0.2, wires=[1])
            MS(0.2, 0.3, wires=[1, 2])
            MS(0.4, 0.5, 0.1, wires=[1, 2])

        dev.apply(tape.operations)

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "native"
        assert dev.job["input"]["qubits"] == 3

        assert len(dev.job["input"]["circuits"][0]["circuit"]) == 4
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "gpi",
            "target": 0,
            "phase": 0.1,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][1] == {
            "gate": "gpi2",
            "target": 1,
            "phase": 0.2,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][2] == {
            "gate": "ms",
            "targets": [1, 2],
            "phases": [0.2, 0.3],
            "angle": 0.25,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][3] == {
            "gate": "ms",
            "targets": [1, 2],
            "phases": [0.4, 0.5],
            "angle": 0.1,
        }

    def test_parameterized_native_op_batch_submit(self, mocker):
        """Tests job attribute for several parameterized native operations with batch_execute."""

        class StopExecute(Exception):
            pass

        def mock_submit_job(*args):
            raise StopExecute()

        mocker.patch("pennylane_ionq.device.SimulatorDevice._submit_job", mock_submit_job)
        dev = SimulatorDevice(wires=(0,), gateset="native")

        with qml.tape.QuantumTape(shots=1024) as tape1:
            GPI(0.7, wires=[0])
            GPI2(0.8, wires=[0])
            qml.expval(qml.PauliZ(0))
        with qml.tape.QuantumTape(shots=1024) as tape2:
            GPI2(0.9, wires=[0])
            qml.expval(qml.PauliZ(0))

        try:
            dev.batch_execute([tape1, tape2])
        except StopExecute:
            pass

        assert dev.job["input"]["format"] == "ionq.circuit.v0"
        assert dev.job["input"]["gateset"] == "native"
        assert dev.job["target"] == "simulator"
        assert dev.job["input"]["qubits"] == 1
        assert len(dev.job["input"]["circuits"]) == 2
        assert dev.job["input"]["circuits"][0]["circuit"][0] == {
            "gate": "gpi",
            "target": 0,
            "phase": 0.7,
        }
        assert dev.job["input"]["circuits"][0]["circuit"][1] == {
            "gate": "gpi2",
            "target": 0,
            "phase": 0.8,
        }
        assert dev.job["input"]["circuits"][1]["circuit"][0] == {
            "gate": "gpi2",
            "target": 0,
            "phase": 0.9,
        }

    @pytest.mark.parametrize(
        "phi0, phi1, theta",
        [
            (0.1, 0.2, 0.25),  # Default fully entangling case
            (0, 0.3, 0.1),  # Partially entangling case
            (1.5, 2.7, 0),  # No entanglement case
        ],
    )
    def test_ms_gate_theta_variation(self, phi0, phi1, theta, tol=1e-6):
        """Test MS gate with different theta values to ensure correct entanglement behavior."""
        ms_gate = MS(phi0, phi1, theta, wires=[0, 1])

        # Compute the matrix representation of the gate
        computed_matrix = ms_gate.compute_matrix(*ms_gate.data)

        # Expected matrix
        cos = np.cos(theta / 2)
        exp = np.exp
        pi = np.pi
        i = 1j
        expected_matrix = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [cos, 0, 0, -i * exp(-2 * pi * i * (phi0 + phi1))],
                    [0, cos, -i * exp(-2 * pi * i * (phi0 - phi1)), 0],
                    [0, -i * exp(2 * pi * i * (phi0 - phi1)), cos, 0],
                    [-i * exp(2 * pi * i * (phi0 + phi1)), 0, 0, cos],
                ]
            )
        )

        assert list(ms_gate.data) == [phi0, phi1, theta]
        assert np.allclose(
            computed_matrix, expected_matrix, atol=tol
        ), "Computed matrix does not match the expected matrix"

    def test_simple_operations_SWAP_gate(self, requires_api):
        """Test SWAP gate operation is correctly processed and sent to IonQ."""
        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires=0)
            qml.SWAP(wires=[0, 1])
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-5
        ), "The IonQ and simulator results do not agree."

    def test_simple_operations_controlled_gate(self, requires_api):
        """Test a controlled gate operation is correctly processed and sent to IonQ."""
        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-5
        ), "The IonQ and simulator results do not agree."
