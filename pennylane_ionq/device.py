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
from typing import List
import warnings
from time import sleep

import numpy as np

from pennylane import pauli_decompose, SparseHamiltonian
from pennylane.devices import QubitDevice
from pennylane.ops.op_math import Exp, Sum, SProd
from pennylane.ops import Identity, PauliX, PauliY, PauliZ
from pennylane.ops.op_math.prod import Prod

from pennylane.measurements import (
    Shots,
)
from pennylane.resource import Resources
from pennylane.ops.op_math.linear_combination import LinearCombination

from .api_client import Job, JobExecutionError
from .exceptions import (
    CircuitIndexNotSetException,
    ComplexEvolutionCoefficientsNotSupported,
    NotSupportedEvolutionInstance,
    OperatorNotSupportedInEvolutionGateGenerator,
)
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
    "Evolution": "pauliexp",
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

PAULI_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}

NO_ANALYTIC_MSG = "The ionq device does not support analytic expectation values."


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
    pennylane_requires = ">=0.43.0"
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires,
        *,
        target="simulator",
        gateset="qis",
        shots=None,
        api_key=None,
        error_mitigation=None,
        sharpen=False,
    ):

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

    def batch_transform(self, circuit):
        """Apply a batch transform for preprocessing a circuit prior to execution."""

        if not circuit.shots:
            raise ValueError(NO_ANALYTIC_MSG)
        return super().batch_transform(circuit)

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
            self.job["target"] = "qpu.aria-1"
            warnings.warn(
                "The ionq_qpu backend is deprecated. Defaulting to ionq_qpu.aria-1.",
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

        for operation in operations:
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
        wires = self.map_wires(operation.wires).tolist()
        if operation.name == "Evolution":
            self._apply_evolution_operation(operation, circuit_index, wires)
        else:
            self._apply_simple_operation(operation, circuit_index, wires)

    def _apply_evolution_operation(self, operation, circuit_index, wires):
        """Applies Evolution operations to the internal device state.
        The number of steps argument for Evolution gate will be ignored even if provided because
        IonQ implements hardware-efficient approximate compilation schemes for pauliexp gates.
        """
        warnings.warn(
            "The 'num_steps' argument for the Evolution gate will be ignored. The API maps this "
            "gate to IonQ's 'pauliexp' gate, for which IonQ implements its own hardware-efficient "
            "approximate compilation schemes.",
            UserWarning,
        )
        name = operation.name
        terms = self._extract_evolution_pauli_terms(operation, wires)
        coefficients = self._extract_evolution_coefficients(operation, wires)
        terms, coefficients = self._remove_trivial_terms(terms, coefficients)
        if len(terms) > 0:
            gate = {"gate": self._operation_map[name]}
            gate["targets"] = wires
            gate["terms"] = terms
            # 1. Float conversion to prevent numpy types (np.float64) in the JSON payload.
            # 2. IonQ API expects positive time values for their `pauliexp` gate.
            gate["time"] = abs(float(operation.param))
            # 3. Add missing sign convention to coefficients by multiplying by -1.
            gate["coefficients"] = [-1 * float(v) for v in coefficients]
            self.input["circuits"][circuit_index]["circuit"].append(gate)

    def _apply_simple_operation(self, operation, circuit_index, wires):
        """Applies regular operations (gates) to the internal device state."""
        name = operation.name
        params = operation.parameters
        gate = {"gate": self._operation_map[name]}
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
            if len(params) > 1:
                gate["phases"] = [float(v) for v in params[:2]]
                if len(params) > 2:
                    gate["angle"] = float(params[2])
            else:
                gate["phase"] = float(params[0])
        elif params:
            gate["rotation"] = float(params[0])
        self.input["circuits"][circuit_index]["circuit"].append(gate)

    def _remove_trivial_terms(self, terms, coefficients):
        """Removes II..I terms from the list of terms."""
        cleaned_up_terms = []
        cleaned_up_coefficients = []
        for i, term in enumerate(terms):
            if not "X" in term and not "Y" in term and not "Z" in term:
                continue
            cleaned_up_terms.append(term)
            cleaned_up_coefficients.append(coefficients[i])
        return cleaned_up_terms, cleaned_up_coefficients

    def _extract_evolution_coefficients(self, operation, wires: List[int]) -> List[float]:
        coefficients = None
        operation_generator = operation.generator()
        if isinstance(operation_generator, LinearCombination):
            coefficients = []
            coeffs = operation_generator.coeffs.tolist()
            operations = operation_generator.ops
            for i, _ in enumerate(operations):
                operation = coeffs[i] * operations[i]
                if isinstance(operations[i], (Sum, Prod, PauliX, PauliY, PauliZ, Identity)):
                    coefficients.extend(operation.terms()[0])
                else:
                    op_wires = operation.wires.tolist()
                    pauli_decomp = pauli_decompose(
                        operation.matrix(), wire_order=op_wires, pauli=False
                    )
                    coefficients.extend(pauli_decomp.coeffs.tolist())
        elif isinstance(operation_generator, SparseHamiltonian):
            dense_matrix = operation_generator.H.toarray()
            linear_combination = pauli_decompose(dense_matrix, wire_order=wires, pauli=False)
            coefficients = linear_combination.coeffs.tolist()
        elif isinstance(operation_generator, SProd):
            if isinstance(operation_generator.base, (PauliX, PauliY, PauliZ, Identity)):
                coefficients = [operation_generator.scalar]
            elif isinstance(operation_generator.base, (Sum, Prod)):
                coefficients = [
                    operation_generator.scalar * float(c)
                    for c in operation_generator.base.terms()[0]
                ]
            elif isinstance(operation_generator.base, Exp):
                # can we do anything smarter here?
                pauli_rep = pauli_decompose(
                    operation_generator.matrix(), wire_order=wires, pauli=False
                )
                coefficients = pauli_rep.coeffs.tolist()

        if any(isinstance(c, complex) for c in coefficients):
            raise ComplexEvolutionCoefficientsNotSupported()

        return coefficients

    def _extract_evolution_pauli_terms(self, operation, wires: List[int]) -> List[str]:
        ops = None
        operation_generator = operation.generator()
        if isinstance(operation_generator, LinearCombination):
            ops = []
            coeffs = operation_generator.coeffs.tolist()
            operations = operation_generator.ops
            for i, _ in enumerate(operations):
                operation = coeffs[i] * operations[i]
                if isinstance(operations[i], (Sum, Prod, PauliX, PauliY, PauliZ, Identity)):
                    ops.extend(operation.terms()[1])
                else:
                    op_wires = operation.wires.tolist()
                    pauli_decomp = pauli_decompose(
                        operation.matrix(), wire_order=op_wires, pauli=False
                    )
                    ops.extend(pauli_decomp.ops)
        elif isinstance(operation_generator, SparseHamiltonian):
            dense_mat = operation_generator.H.toarray()
            pauli_rep = pauli_decompose(dense_mat, wire_order=wires, pauli=False)
            ops = pauli_rep.ops
        elif isinstance(operation_generator, SProd):
            if isinstance(operation_generator.base, (PauliX, PauliY, PauliZ, Identity)):
                ops = [operation_generator.base]
            elif isinstance(operation_generator.base, (Sum, Prod)):
                ops = operation_generator.base.terms()[1]
            elif isinstance(operation_generator.base, Exp):
                # can we do anything smarter here?
                pauli_rep = pauli_decompose(
                    operation_generator.matrix(), wire_order=wires, pauli=False
                )
                ops = pauli_rep.ops

        if ops is None:
            raise NotSupportedEvolutionInstance()

        return self._operations_to_ionq_pauli_names(ops, wires)

    def _operations_to_ionq_pauli_names(self, ops, wires) -> List[str]:
        """Converts a list of operations to a list of IonQ compatible Pauli matrix names."""

        def map_operand_to_term(operand):
            try:
                return PAULI_MAP[operand.name]
            except KeyError as exc:
                supported = ", ".join(PAULI_MAP.keys())
                raise KeyError(
                    f"Operand {operand.name} is not supported for Evolution gate. "
                    f"Supported operands: {supported}."
                ) from exc

        def join_terms(terms, wires):
            """Pennylane uses big-endian ordering, IonQ uses little-endian ordering."""
            big_endian_term = "".join(terms.get(wire, "I") for wire in wires)
            little_endian_term = big_endian_term[::-1]
            return little_endian_term

        ionq_terms = []
        for op in ops:
            terms = {}
            if isinstance(op, Prod):
                for operand in op.operands:
                    term_name = map_operand_to_term(operand)
                    term_wire = operand.wires[0]
                    terms[term_wire] = term_name
                ionq_terms.append(join_terms(terms, wires))
            elif isinstance(op, (PauliX, PauliY, PauliZ)):
                term_name = map_operand_to_term(op)
                term_wire = op.wires[0]
                terms[term_wire] = term_name
                ionq_terms.append(join_terms(terms, wires))
            elif isinstance(op, Identity):
                ionq_terms.append(join_terms(terms, wires))
            else:
                raise OperatorNotSupportedInEvolutionGateGenerator(
                    f"Unsupported operator in generator of Evolution gate: {op}"
                )
        return ionq_terms

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

    def __init__(self, wires, *, gateset="qis", shots=None, api_key=None):
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
        backend (str): Optional specifier for an IonQ backend. Can be ``"aria-1"``, ``"aria-2"``, etc.
            Default to ``aria-1``.
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires,
        *,
        gateset="qis",
        shots=None,
        backend="aria-1",
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
