# Copyright 2020 Xanadu Quantum Technologies Inc.

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
import itertools
import functools
from time import sleep

import numpy as np

from pennylane import QubitDevice, DeviceError

from .api_client import Job
from ._version import __version__


class IonQDevice(QubitDevice):
    r"""IonQ device for PennyLane.

    Args:
        target (str): the target device, either ``"simulator"`` or ``"qpu"``
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
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

    _operation_map = {
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


    def __init__(self, wires, *, target="simulator", shots=1024, api_key=None):
        if shots is None:
            raise ValueError(
                "The ionq device does not support analytic expectation values"
            )

        super().__init__(wires=wires, shots=shots)
        self.target = target
        self.api_key = api_key
        self.reset()

    def reset(self):
        """Reset the device"""
        self.prob = None
        self.circuit = {"qubits": self.num_wires, "circuit": []}
        self.job = {"lang": "json", "body": self.circuit, "target": self.target}

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operations, **kwargs):
        rotations = kwargs.pop("rotations", [])

        for i, operation in enumerate(operations):
            if i > 0 and operation.name in {"BasisState", "QubitStateVector"}:
                raise DeviceError(
                    "The operation {} is only supported at the beginning of a circuit.".format(
                        operation.name
                    )
                )
            self._apply_operation(operation)

        # diagonalize observables
        for operation in rotations:
            self._apply_operation(operation)

        self._submit_job()

    def _apply_operation(self, operation):
        name = self._operation_map[operation.name]
        wires = self.map_wires(operation.wires).tolist()
        gate = {"gate": name}
        par = operation.parameters

        if len(wires) == 2:
            if name in {"SWAP", "XX", "YY", "ZZ"}:
                # these gates takes two targets
                gate["targets"] = wires
            else:
                gate["control"] = wires[0]
                gate["target"] = wires[1]
        else:
            gate["target"] = wires[0]

        if par:
            gate["rotation"] = par[0]

        self.circuit["circuit"].append(gate)

    def _submit_job(self):
        job = Job(api_key=self.api_key)

        # send job for exection
        job.manager.create(**self.job)

        # retrieve results
        while not job.is_complete:
            sleep(0.01)
            job.reload()

        job.manager.get(job.id.value)

        histogram = job.data.value["histogram"]
        self.prob = np.zeros([2 ** self.num_wires])
        self.prob[np.array([int(i) for i in histogram.keys()])] = list(histogram.values())

    def probability(self, wires=None, shot_range=None, bin_size=None):
        wires = wires or self.wires

        if shot_range is None and bin_size is None:
        	return self.marginal_prob(self.prob, wires)

        return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

    def generate_samples(self):
        number_of_states = 2 ** self.num_wires
        samples = self.sample_basis_states(number_of_states, self.prob)
        return QubitDevice.states_to_binary(samples, self.num_wires)


class SimulatorDevice(IonQDevice):
    r"""Simulator device for IonQ.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
    """
    name = "IonQ Simulator PennyLane plugin"
    short_name = "ionq.simulator"

    def __init__(self, wires, *, shots=1024, api_key=None):
        super().__init__(wires=wires, target="simulator", shots=shots, api_key=api_key)


class QPUDevice(IonQDevice):
    r"""QPU device for IonQ.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        api_key (str): The IonQ API key. If not provided, the environment
            variable ``IONQ_API_KEY`` is used.
    """
    name = "IonQ QPU PennyLane plugin"
    short_name = "ionq.qpu"

    def __init__(self, wires, *, shots=1024):
        super().__init__(wires=wires, target="qpu", shots=shots, api_key=api_key)
