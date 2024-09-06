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
import warnings
from time import sleep

import numpy as np

from pennylane import QubitDevice, DeviceError

from .api_client import Job, JobExecutionError
from ._version import __version__

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
    pennylane_requires = ">=0.15.0"
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
        self.target = target
        self.api_key = api_key
        self.gateset = gateset
        self.error_mitigation = error_mitigation
        self.sharpen = sharpen
        self._operation_map = _GATESET_OPS[gateset]
        self.reset()

    def reset(self):
        """Reset the device"""
        self._prob_array = None
        self.histogram = None
        self.circuit = {
            "format": "ionq.circuit.v0",
            "qubits": self.num_wires,
            "circuit": [],
            "gateset": self.gateset,
        }
        self.job = {
            "input": self.circuit,
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

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operations, **kwargs):
        self.reset()
        rotations = kwargs.pop("rotations", [])

        if len(operations) == 0 and len(rotations) == 0:
            warnings.warn("Circuit is empty. Empty circuits return failures. Submitting anyway.")

        for i, operation in enumerate(operations):
            if i > 0 and operation.name in {
                "BasisState",
                "QubitStateVector",
                "StatePrep",
            }:
                raise DeviceError(
                    f"The operation {operation.name} is only supported at the beginning of a circuit."
                )
            self._apply_operation(operation)

        # diagonalize observables
        for operation in rotations:
            self._apply_operation(operation)

        self._submit_job()

    def _apply_operation(self, operation):
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

        self.circuit["circuit"].append(gate)

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
        self.histogram = job.data.value

    @property
    def prob(self):
        """None or array[float]: Array of computational basis state probabilities. If
        no job has been submitted, returns ``None``.
        """
        if self.histogram is None:
            return None

        if self._prob_array is None:
            # The IonQ API returns basis states using little-endian ordering.
            # Here, we rearrange the states to match the big-endian ordering
            # expected by PennyLane.
            basis_states = (
                int(bin(int(k))[2:].rjust(self.num_wires, "0")[::-1], 2) for k in self.histogram
            )
            idx = np.fromiter(basis_states, dtype=int)

            # convert the sparse probs into a probability array
            self._prob_array = np.zeros([2**self.num_wires])

            # histogram values don't always perfectly sum to exactly one
            histogram_values = self.histogram.values()
            norm = sum(histogram_values)
            self._prob_array[idx] = np.fromiter(histogram_values, float) / norm

        return self._prob_array

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
