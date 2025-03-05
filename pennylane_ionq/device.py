# Copyright 2019-2021 Xanadu Quantum Technologies Inc.

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
This module contains the device class for constructing IonQ devices for PennyLane.
"""

# pylint: disable=too-many-arguments

import inspect
import logging
import warnings
from time import sleep

import numpy as np

from pennylane.devices import QubitDevice

from pennylane.measurements import (
    Shots,
)
from pennylane.resource import Resources

from .api_client import Job, JobExecutionError
from ._version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_qis_operation_map = {
    # native PennyLane operations also native to IonQ
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "CNOT": "cnot",
    "SWAP": "swap",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "S": "s",
    "S.inv": "si",
    "T": "t",
    "T.inv": "ti",
    "SX": "v",
    "SX.inv": "vi",
    # additional operations not native to PennyLane but present in IonQ
    "XX": "xx",
    "YY": "yy",
    "ZZ": "zz",
}

_native_operation_map = {
    "GPI": "gpi",
    "GPI2": "gpi2",
    "MS": "ms",
}

_GATESET_OPS = {
    "native": _native_operation_map,
    "qis": _qis_operation_map,
}


class CircuitIndexNotSetException(Exception):
    """Raised when after submitting multiple circuits circuit index is not set
    before the user want to access implementation methods of IonQDevice
    like probability(), estimate_probability(), sample() or the prob property.
    """

    def __init__(self):
        self.message = (
            "Because multiple circuits have been submitted in this job, the index of the circuit "
            "you want to access must be first set via the set_current_circuit_index device method."
        )
        super().__init__(self.message)


class IonQDevice(QubitDevice):
    r"""IonQ device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).

    Kwargs:
        target (str): the target device, either ``"simulator"`` or ``"qpu"``. Defaults to ``simulator``.
        gateset (str): the target gateset, either ``"qis"`` or ``"native"``. Defaults to ``qis``.
        shots (int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1024.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
        error_mitigation (dict): settings for error mitigation when creating a job. Defaults to None.
            Not available on all backends. Set by default on some hardware systems. See
            `IonQ API Job Creation <https://docs.ionq.com/#tag/jobs/operation/createJob>`_  and
            `IonQ Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`_ for details.
            Valid keys include: ``debias`` (bool).
        sharpen (bool): whether to use sharpening when accessing the results of an executed job. Defaults to None
            (no value passed at job retrieval). Will generally return more accurate results if your expected output
            distribution has peaks. See `IonQ Debiasing and Sharpening
            <https://ionq.com/resources/debiasing-and-sharpening>`_ for details.
    """

    # pylint: disable=too-many-instance-attributes
    name = "IonQ PennyLane plugin"
    short_name = "ionq"
    pennylane_requires = ">=0.38.0"
    version = __version__
    author = "Xanadu Inc."

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

    # Note: unlike QubitDevice, IonQ does not support QubitUnitary,
    # and therefore does not support the Hermitian observable.
    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Identity", "Prod"}

    def __init__(
        self,
        wires,
        *,
        target="simulator",
        gateset="qis",
        shots=1024,
        api_key=None,
        error_mitigation=None,
        sharpen=False,
    ):
        if shots is None:
            raise ValueError("The ionq device does not support analytic expectation values.")

        super().__init__(wires=wires, shots=shots)
        self._current_circuit_index = None
        self.target = target
        self.api_key = api_key
        self.gateset = gateset
        self.error_mitigation = error_mitigation
        self.sharpen = sharpen
        self._operation_map = _GATESET_OPS[gateset]
        self.histograms = []
        self._samples = None
        self.reset()

    def reset(self, circuits_array_length=1):
        """Reset the device"""
        self._current_circuit_index = None
        self._samples = None
        self.histograms = []
        self.input = {
            "format": "ionq.circuit.v0",
            "qubits": self.num_wires,
            "circuits": [{"circuit": []} for _ in range(circuits_array_length)],
            "gateset": self.gateset,
        }
        self.job = {
            "input": self.input,
            "target": self.target,
            "shots": self.shots,
        }
        if self.error_mitigation is not None:
            self.job["error_mitigation"] = self.error_mitigation
        if self.job["target"] == "qpu":
            self.job["target"] = "qpu.harmony"
            warnings.warn(
                "The ionq_qpu backend is deprecated. Defaulting to ionq_qpu.harmony.",
                UserWarning,
                stacklevel=2,
            )

    def set_current_circuit_index(self, circuit_index):
        """Sets the index of the current circuit for which operations are applied upon.
        In case of multiple circuits being submitted via batch_execute method
        self._current_circuit_index tracks the index of the current circuit.
        """
        self._current_circuit_index = circuit_index

    def batch_execute(self, circuits):
        """Execute a batch of quantum circuits on the device.

        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.

        Args:
            circuits (list[~.tape.QuantumTape]): circuits to execute on the device

        Returns:
            list[array[float]]: list of measured value(s)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(  # pragma: no cover
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        self.reset(circuits_array_length=len(circuits))

        for circuit_index, circuit in enumerate(circuits):
            self.check_validity(circuit.operations, circuit.observables)
            self.batch_apply(
                circuit.operations,
                rotations=self._get_diagonalizing_gates(circuit),
                circuit_index=circuit_index,
            )

        self._submit_job()

        results = []
        for circuit_index, circuit in enumerate(circuits):
            self.set_current_circuit_index(circuit_index)
            self._samples = self.generate_samples()

            # compute the required statistics
            if self._shot_vector is not None:
                result = self.shot_vec_statistics(circuit)
            else:
                result = self.statistics(circuit)
                single_measurement = len(circuit.measurements) == 1

                result = result[0] if single_measurement else tuple(result)

            self.set_current_circuit_index(None)
            self._samples = None
            results.append(result)

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        if self.tracker.active:
            for circuit in circuits:
                shots_from_dev = self._shots if not self.shot_vector else self._raw_shot_sequence
                tape_resources = circuit.specs["resources"]

                resources = Resources(  # temporary until shots get updated on tape !
                    tape_resources.num_wires,
                    tape_resources.num_gates,
                    tape_resources.gate_types,
                    tape_resources.gate_sizes,
                    tape_resources.depth,
                    Shots(shots_from_dev),
                )
                self.tracker.update(
                    executions=1,
                    shots=self._shots,
                    results=results,
                    resources=resources,
                )

            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results

    def batch_apply(self, operations, circuit_index, **kwargs):
        "Apply circuit operations when submitting for execution a batch of circuits."

        rotations = kwargs.pop("rotations", [])

        if len(operations) == 0 and len(rotations) == 0:
            warnings.warn("Circuit is empty. Empty circuits return failures. Submitting anyway.")

        for i, operation in enumerate(operations):
            self._apply_operation(operation, circuit_index)

        # diagonalize observables
        for operation in rotations:
            self._apply_operation(operation, circuit_index)

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operations, **kwargs):
        """Implementation of QubitDevice abstract method apply."""

        self.reset()
        rotations = kwargs.pop("rotations", [])

        if len(operations) == 0 and len(rotations) == 0:
            warnings.warn("Circuit is empty. Empty circuits return failures. Submitting anyway.")

        for i, operation in enumerate(operations):
            self._apply_operation(operation)

        # diagonalize observables
        for operation in rotations:
            self._apply_operation(operation)

        self._submit_job()

    def _apply_operation(self, operation, circuit_index=0):
        """Applies operations to the internal device state.

        Args:
            operation (.Operation): operation to apply on the device
            circuit_index: index of the circuit to apply operation to
        """
        name = operation.name
        wires = self.map_wires(operation.wires).tolist()
        gate = {"gate": self._operation_map[name]}
        par = operation.parameters

        if len(wires) == 2:
            if name in {"SWAP", "XX", "YY", "ZZ", "MS"}:
                # these gates takes two targets
                gate["targets"] = wires
            else:
                gate["control"] = wires[0]
                gate["target"] = wires[1]
        else:
            gate["target"] = wires[0]

        if self.gateset == "native":
            if len(par) > 1:
                gate["phases"] = [float(v) for v in par[:2]]
                if len(par) > 2:
                    gate["angle"] = float(par[2])
            else:
                gate["phase"] = float(par[0])
        elif par:
            gate["rotation"] = float(par[0])

        self.input["circuits"][circuit_index]["circuit"].append(gate)

    def _submit_job(self):

        job = Job(api_key=self.api_key)

        # send job for exection
        job.manager.create(**self.job)

        # retrieve results
        while not job.is_complete:
            sleep(0.01)
            job.reload()
            if job.is_failed:
                raise JobExecutionError("Job failed")

        params = {} if self.sharpen is None else {"sharpen": self.sharpen}

        job.manager.get(resource_id=job.id.value, params=params)

        # The returned job histogram is of the form
        # dict[str, float], and maps the computational basis
        # state (as a base-10 integer string) to the probability
        # as a floating point value between 0 and 1.
        # e.g., {"0": 0.413, "9": 0.111, "17": 0.476}
        some_inner_value = next(iter(job.data.value.values()))
        if isinstance(some_inner_value, dict):
            self.histograms = []
            for key in job.data.value.keys():
                self.histograms.append(job.data.value[key])
        else:
            self.histograms = []
            self.histograms.append(job.data.value)

    @property
    def prob(self):
        """None or array[float]: Array of computational basis state probabilities. If
        no job has been submitted, returns ``None``.
        """
        if self._current_circuit_index is None and len(self.histograms) > 1:
            raise CircuitIndexNotSetException()

        if self._current_circuit_index is not None:
            histogram = self.histograms[self._current_circuit_index]
        else:
            try:
                histogram = self.histograms[0]
            except IndexError:
                return None

        # The IonQ API returns basis states using little-endian ordering.
        # Here, we rearrange the states to match the big-endian ordering
        # expected by PennyLane.
        basis_states = (int(bin(int(k))[2:].rjust(self.num_wires, "0")[::-1], 2) for k in histogram)
        idx = np.fromiter(basis_states, dtype=int)

        # convert the sparse probs into a probability array
        prob_array = np.zeros([2**self.num_wires])

        # histogram values don't always perfectly sum to exactly one
        histogram_values = histogram.values()
        norm = sum(histogram_values)
        prob_array[idx] = np.fromiter(histogram_values, float) / norm

        return prob_array

    def probability(self, wires=None, shot_range=None, bin_size=None):
        wires = wires or self.wires

        if shot_range is None and bin_size is None:
            return self.marginal_prob(self.prob, wires)

        return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)


class SimulatorDevice(IonQDevice):
    r"""Simulator device for IonQ.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        gateset (str): the target gateset, either ``"qis"`` or ``"native"``. Defaults to ``qis``.
        shots (int, list[int], None): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
            Defaults to 1024.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
    """

    name = "IonQ Simulator PennyLane plugin"
    short_name = "ionq.simulator"

    def __init__(self, wires, *, gateset="qis", shots=1024, api_key=None):
        super().__init__(
            wires=wires,
            target="simulator",
            gateset=gateset,
            shots=shots,
            api_key=api_key,
        )

    def generate_samples(self):
        """Generates samples by random sampling with the probabilities returned by the simulator."""
        number_of_states = 2**self.num_wires
        samples = self.sample_basis_states(number_of_states, self.prob)
        return QubitDevice.states_to_binary(samples, self.num_wires)


class QPUDevice(IonQDevice):
    r"""QPU device for IonQ.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        gateset (str): the target gateset, either ``"qis"`` or ``"native"``. Defaults to ``qis``.
        backend (str): Optional specifier for an IonQ backend. Can be ``"harmony"``, ``"aria-1"``, etc.
            Default to ``harmony``.
        shots (int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1024. If a list of integers is passed, the
            circuit evaluations are batched over the list of shots.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
        error_mitigation (dict): settings for error mitigation when creating a job. Defaults to None.
            Not available on all backends. Set by default on some hardware systems. See
            `IonQ API Job Creation <https://docs.ionq.com/#tag/jobs/operation/createJob>`_  and
            `IonQ Debiasing and Sharpening <https://ionq.com/resources/debiasing-and-sharpening>`_ for details.
            Valid keys include: ``debias`` (bool).
        sharpen (bool): whether to use sharpening when accessing the results of an executed job.
            Defaults to None (no value passed at job retrieval). Will generally return more accurate results if
            your expected output distribution has peaks. See `IonQ Debiasing and Sharpening
            <https://ionq.com/resources/debiasing-and-sharpening>`_ for details.
    """

    name = "IonQ QPU PennyLane plugin"
    short_name = "ionq.qpu"

    def __init__(
        self,
        wires,
        *,
        gateset="qis",
        shots=1024,
        backend="harmony",
        error_mitigation=None,
        sharpen=None,
        api_key=None,
    ):
        target = "qpu"
        self.backend = backend
        if self.backend is not None:
            target += "." + self.backend
        super().__init__(
            wires=wires,
            target=target,
            gateset=gateset,
            shots=shots,
            api_key=api_key,
            error_mitigation=error_mitigation,
            sharpen=sharpen,
        )

    def generate_samples(self):
        """Generates samples from the qpu.

        Note that the order of the samples returned here is not indicative of the order in which
        the experiments were done, but is instead controlled by a random shuffle (and hence
        set by numpy random seed).
        """
        number_of_states = 2**self.num_wires
        counts = np.rint(
            self.prob * self.shots,
            out=np.zeros(number_of_states, dtype=int),
            casting="unsafe",
        )
        samples = np.repeat(np.arange(number_of_states), counts)
        np.random.shuffle(samples)
        return QubitDevice.states_to_binary(samples, self.num_wires)
