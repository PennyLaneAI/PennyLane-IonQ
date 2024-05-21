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
import inspect
import logging
import warnings
from time import sleep
from typing import Union

import numpy as np
import pennylane as qml

from pennylane import QubitDevice, DeviceError

from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    MeasurementTransform,
    MeasurementValue,
    MutualInfoMP,
    ProbabilityMP,
    SampleMeasurement,
    SampleMP,
    ShadowExpvalMP,
    Shots,
    StateMeasurement,
    StateMP,
    VarianceMP,
    VnEntropyMP
)
from pennylane.resource import Resources
from pennylane.tape import QuantumTape
from pennylane.wires import Wires

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
            raise ValueError(
                "The ionq device does not support analytic expectation values."
            )

        super().__init__(wires=wires, shots=shots)
        self.target = target
        self.api_key = api_key
        self.gateset = gateset
        self.error_mitigation = error_mitigation
        self.sharpen = sharpen
        self._operation_map = _GATESET_OPS[gateset]
        self.histogram = None
        self.histograms = None
        self._samples = None
        self._samples_array = []
        self.reset()

    def reset(self, circuits_array_length=0):
        """Reset the device"""
        self.histogram = None
        self.histograms = None
        self._samples = None
        self._samples_array = []
        if circuits_array_length == 0:
            self.input = {
                "format": "ionq.circuit.v0",
                "qubits": self.num_wires,
                "circuit": [],
                "gateset": self.gateset,
            }
        else:
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
            logger.debug(
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i)
                    for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
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
            sample_type = (SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP)
            if self.shots is not None or any(
                isinstance(m, sample_type) for m in circuit.measurements
            ):
                self._samples_array.append(self.generate_samples(circuit_index))

            # compute the required statistics
            if self._shot_vector is not None:
                result = self.shot_vec_statistics(circuit)
            else:
                result = self.statistics(circuit, circuit_index=circuit_index)
                single_measurement = len(circuit.measurements) == 1

                result = result[0] if single_measurement else tuple(result)

            results.append(result)

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        if self.tracker.active:
            for circuit in circuits:
                shots_from_dev = (
                    self._shots if not self.shot_vector else self._raw_shot_sequence
                )
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

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def batch_apply(self, operations, circuit_index=None, **kwargs):

        rotations = kwargs.pop("rotations", [])

        if len(operations) == 0 and len(rotations) == 0:
            warnings.warn(
                "Circuit is empty. Empty circuits return failures. Submitting anyway."
            )

        for i, operation in enumerate(operations):
            if i > 0 and operation.name in {
                "BasisState",
                "QubitStateVector",
                "StatePrep",
            }:
                raise DeviceError(
                    f"The operation {operation.name} is only supported at the beginning of a circuit."
                )
            self._apply_operation(operation, circuit_index)

        # diagonalize observables
        for operation in rotations:
            self._apply_operation(operation, circuit_index)

    def apply(self, operations, **kwargs):
        self.reset()
        rotations = kwargs.pop("rotations", [])

        if len(operations) == 0 and len(rotations) == 0:
            warnings.warn(
                "Circuit is empty. Empty circuits return failures. Submitting anyway."
            )

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

    def _apply_operation(self, operation, circuit_index=None):
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
                gate["phases"] = [float(v) for v in par]
            else:
                gate["phase"] = float(par[0])
        elif par:
            gate["rotation"] = float(par[0])

        if "circuit" in self.input:
            self.input["circuit"].append(gate)
        else:
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
            self.histogram = job.data.value

    def _measure(
        self,
        measurement: Union[SampleMeasurement, StateMeasurement],
        shot_range=None,
        bin_size=None,
        circuit_index=None,
    ):
        """Compute the corresponding measurement process depending on ``shots`` and the measurement
        type.

        Args:
            measurement (Union[SampleMeasurement, StateMeasurement]): measurement process
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Raises:
            ValueError: if the measurement cannot be computed

        Returns:
            Union[float, dict, list[float]]: result of the measurement
        """
        if self.shots is None:
            if isinstance(measurement, StateMeasurement):
                return measurement.process_state(
                    state=self.state, wire_order=self.wires
                )

            raise ValueError(
                "Shots must be specified in the device to compute the measurement "
                f"{measurement.__class__.__name__}"
            )
        if isinstance(measurement, StateMeasurement):
            warnings.warn(
                f"Requested measurement {measurement.__class__.__name__} with finite shots; the "
                "returned state information is analytic and is unaffected by sampling. "
                "To silence this warning, set shots=None on the device.",
                UserWarning,
            )
            return measurement.process_state(state=self.state, wire_order=self.wires)
        if circuit_index is None:
            return measurement.process_samples(
                samples=self._samples,
                wire_order=self.wires,
                shot_range=shot_range,
                bin_size=bin_size,
            )
        else:
            return measurement.process_samples(
                samples=self._samples_array[circuit_index],
                wire_order=self.wires,
                shot_range=shot_range,
                bin_size=bin_size,
            )

    def prob(self, circuit_index=None):
        """None or array[float]: Array of computational basis state probabilities. If
        no job has been submitted, returns ``None``.
        """
        if self.histogram is None and self.histograms is None:
            return None

        if self.histogram is not None:
            histogram = self.histogram
        else:
            histogram = self.histograms[circuit_index]

        # The IonQ API returns basis states using little-endian ordering.
        # Here, we rearrange the states to match the big-endian ordering
        # expected by PennyLane.
        basis_states = (
            int(bin(int(k))[2:].rjust(self.num_wires, "0")[::-1], 2) for k in histogram
        )
        idx = np.fromiter(basis_states, dtype=int)

        # convert the sparse probs into a probability array
        prob_array = np.zeros([2**self.num_wires])

        # histogram values don't always perfectly sum to exactly one
        histogram_values = histogram.values()
        norm = sum(histogram_values)
        prob_array[idx] = np.fromiter(histogram_values, float) / norm

        return prob_array

    def probability(
        self, wires=None, shot_range=None, bin_size=None, circuit_index=None
    ):
        wires = wires or self.wires

        if shot_range is None and bin_size is None:
            return self.marginal_prob(self.prob(circuit_index), wires)

        return self.estimate_probability(
            wires=wires, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index
        )

    def statistics(
        self, circuit: QuantumTape, shot_range=None, bin_size=None, circuit_index=None
    ):  # pylint: disable=too-many-statements
        """Process measurement results from circuit execution and return statistics.

        This includes returning expectation values, variance, samples, probabilities, states, and
        density matrices. Overwrites the method in QubiDevice class to accomodate batch submission
        of circuits. Please check the base class for explanatory comments on how this function 
        should be working.

        """
        if circuit_index is None:
            return super().statistics(circuit, shot_range=shot_range, bin_size=bin_size)
    
        measurements = circuit.measurements
        results = []

        for m in measurements:
            # TODO: Remove this when all overriden measurements support the `MeasurementProcess` class
            if isinstance(m.mv, list):
                # MeasurementProcess stores information needed for processing if terminal measurement
                # uses a list of mid-circuit measurement values
                obs = m
            else:
                obs = m.obs or m.mv or m
            # Check if there is an overriden version of the measurement process
            if method := getattr(self, self.measurement_map[type(m)], False):
                if isinstance(m, MeasurementTransform):
                    result = method(tape=circuit)
                else:
                    result = method(m, shot_range=shot_range, bin_size=bin_size)
            # 1. Based on the measurement type, compute statistics
            # Pass instances directly
            elif isinstance(m, ExpectationMP):
                result = self.expval(obs, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)

            elif isinstance(m, VarianceMP):
                result = self.var(obs, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)

            elif isinstance(m, SampleMP):
                samples = self.sample(obs, shot_range=shot_range, bin_size=bin_size, counts=False, circuit_index=circuit_index)
                result = self._asarray(qml.math.squeeze(samples))

            elif isinstance(m, CountsMP):
                result = self.sample(m, shot_range=shot_range, bin_size=bin_size, counts=True, circuit_index=circuit_index)

            elif isinstance(m, ProbabilityMP):
                result = self.probability(wires=m.wires, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)

            elif isinstance(m, StateMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError(
                        "The state or density matrix cannot be returned in combination "
                        "with other return types"
                    )

                if self.shots is not None:
                    warnings.warn(
                        "Requested state or density matrix with finite shots; the returned "
                        "state information is analytic and is unaffected by sampling. To silence "
                        "this warning, set shots=None on the device.",
                        UserWarning,
                    )

                # Check if the state is accessible and decide to return the state or the density
                # matrix.
                state = self.access_state(wires=obs.wires)
                result = self._asarray(state, dtype=self.C_DTYPE)

            elif isinstance(m, VnEntropyMP):
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError(
                        "Returning the Von Neumann entropy is not supported when using custom wire labels"
                    )

                if self._shot_vector is not None:
                    raise NotImplementedError(
                        "Returning the Von Neumann entropy is not supported with shot vectors."
                    )

                if self.shots is not None:
                    warnings.warn(
                        "Requested Von Neumann entropy with finite shots; the returned "
                        "result is analytic and is unaffected by sampling. To silence "
                        "this warning, set shots=None on the device.",
                        UserWarning,
                    )
                result = self.vn_entropy(wires=obs.wires, log_base=obs.log_base)

            elif isinstance(m, MutualInfoMP):
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError(
                        "Returning the mutual information is not supported when using custom wire labels"
                    )

                if self._shot_vector is not None:
                    raise NotImplementedError(
                        "Returning the mutual information is not supported with shot vectors."
                    )

                if self.shots is not None:
                    warnings.warn(
                        "Requested mutual information with finite shots; the returned "
                        "state information is analytic and is unaffected by sampling. To silence "
                        "this warning, set shots=None on the device.",
                        UserWarning,
                    )
                wires0, wires1 = obs.raw_wires
                result = self.mutual_info(wires0=wires0, wires1=wires1, log_base=obs.log_base)

            elif isinstance(m, ClassicalShadowMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError(
                        "Classical shadows cannot be returned in combination "
                        "with other return types"
                    )
                result = self.classical_shadow(obs, circuit)

            elif isinstance(m, ShadowExpvalMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError(
                        "Classical shadows cannot be returned in combination "
                        "with other return types"
                    )
                result = self.shadow_expval(obs, circuit=circuit)

            elif isinstance(m, MeasurementTransform):
                result = m.process(tape=circuit, device=self)

            elif isinstance(m, (SampleMeasurement, StateMeasurement)):
                result = self._measure(m, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)

            elif m.return_type is not None:
                name = obs.name if isinstance(obs, qml.operation.Operator) else type(obs).__name__
                raise qml.QuantumFunctionError(
                    f"Unsupported return type specified for observable {name}"
                )
            else:
                result = None

            # 2. Post-process statistics results (if need be)
            if isinstance(
                m,
                (
                    ExpectationMP,
                    VarianceMP,
                    ProbabilityMP,
                    VnEntropyMP,
                    MutualInfoMP,
                    ShadowExpvalMP,
                ),
            ):
                # Result is a float
                result = self._asarray(result, dtype=self.R_DTYPE)

            if self._shot_vector is not None and isinstance(result, np.ndarray):
                # In the shot vector case, measurement results may be of shape (N, 1) instead of (N,)
                # Squeeze the result to transform the results
                #
                # E.g.,
                # before:
                # [[0.489]
                #  [0.511]
                #  [0.   ]
                #  [0.   ]]
                #
                # after: [0.489 0.511 0.    0.   ]
                result = qml.math.squeeze(result)

            # 3. Append to final list
            if result is not None:
                results.append(result)

        return results

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None, circuit_index=None):
        """Return the estimated probability of each computational basis state
        using the generated samples. Overwrites the method in QubiDevice class 
        to accomodate batch submission of circuits. 

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """

        wires = wires or self.wires
        # convert to a Wires object
        wires = Wires(wires)
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)

        if circuit_index is None:
            stored_samples_prop = self._samples
        else:
            stored_samples_prop = self._samples_array[circuit_index]

        if shot_range is None:
            # The Ellipsis (...) corresponds to broadcasting and shots dimensions or only shots
            samples = stored_samples_prop[..., device_wires]
        else:
            # The Ellipsis (...) corresponds to the broadcasting dimension or no axis at all
            samples = stored_samples_prop[..., slice(*shot_range), device_wires]

        # convert samples from a list of 0, 1 integers, to base 10 representation
        powers_of_two = 2 ** np.arange(num_wires)[::-1]
        indices = samples @ powers_of_two

        # `self._samples` typically has two axes ((shots, wires)) but can also have three with
        # broadcasting ((batch_size, shots, wires)) so that we simply read out the batch_size.
        batch_size = stored_samples_prop.shape[0] if np.ndim(stored_samples_prop) == 3 else None
        dim = 2**num_wires
        # count the basis state occurrences, and construct the probability vector
        if bin_size is not None:
            num_bins = samples.shape[-2] // bin_size
            prob = self._count_binned_samples(indices, batch_size, dim, bin_size, num_bins)
        else:
            prob = self._count_unbinned_samples(indices, batch_size, dim)

        return self._asarray(prob, dtype=self.R_DTYPE)

    def sample(self, observable, shot_range=None, bin_size=None, counts=False, circuit_index=None):
        """Return samples of an observable.

        Overwrites the method in QubiDevice class to accomodate batch submission
        of circuits. 

        Args:
            observable (Observable): the observable to sample
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.
            counts (bool): whether counts (``True``) or raw samples (``False``)
                should be returned

        Raises:
            EigvalsUndefinedError: if no information is available about the
                eigenvalues of the observable

        Returns:
            Union[array[float], dict, list[dict]]: samples in an array of
            dimension ``(shots,)`` or counts
        """
        mp = observable
        no_observable_provided = False
        if isinstance(mp, MeasurementProcess):
            if mp.obs is not None:
                observable = mp.obs
            elif mp.mv is not None and isinstance(mp.mv, MeasurementValue):
                observable = mp.mv
            else:
                no_observable_provided = True

        # translate to wire labels used by device. observable is list when measuring sequence
        # of multiple MeasurementValues
        device_wires = self.map_wires(observable.wires)
        name = None if no_observable_provided else observable.name
        # Select the samples from self._samples that correspond to ``shot_range`` if provided
        if shot_range is None:
            if circuit_index is None:
                sub_samples = self._samples
            else: 
                sub_samples = self._samples_array[circuit_index]
        else:
            # Indexing corresponds to: (potential broadcasting, shots, wires). Note that the last
            # colon (:) is required because shots is the second-to-last axis and the
            # Ellipsis (...) otherwise would take up broadcasting and shots axes.
            if circuit_index is None:
                sub_samples = self._samples[..., slice(*shot_range), :]
            else:
                sub_samples = self._samples_array[circuit_index][..., slice(*shot_range), :]

        if isinstance(name, str) and name in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            # Process samples for observables with eigenvalues {1, -1}
            samples = 1 - 2 * sub_samples[..., device_wires[0]]

        elif no_observable_provided:
            # if no observable was provided then return the raw samples
            if len(observable.wires) != 0:
                # if wires are provided, then we only return samples from those wires
                samples = sub_samples[..., np.array(device_wires)]
            else:
                samples = sub_samples

        else:
            # Replace the basis state in the computational basis with the correct eigenvalue.
            # Extract only the columns of the basis samples required based on ``wires``.
            samples = sub_samples[
                ..., np.array(device_wires)
            ]  # Add np.array here for Jax support.
            powers_of_two = 2 ** np.arange(samples.shape[-1])[::-1]
            indices = samples @ powers_of_two
            indices = np.array(indices)  # Add np.array here for Jax support.
            if isinstance(observable, MeasurementValue):
                eigvals = self._asarray(
                    [observable[i] for i in range(2 ** len(observable.measurements))],
                    dtype=self.R_DTYPE,
                )
                samples = eigvals[indices]
            else:
                try:
                    samples = observable.eigvals()[indices]
                except qml.operation.EigvalsUndefinedError as e:
                    # if observable has no info on eigenvalues, we cannot return this measurement
                    raise qml.operation.EigvalsUndefinedError(
                        f"Cannot compute samples of {observable.name}."
                    ) from e

        num_wires = len(device_wires) if len(device_wires) > 0 else self.num_wires
        if bin_size is None:
            if counts:
                return self._samples_to_counts(samples, mp, num_wires)
            return samples

        if counts:
            shape = (
                (-1, bin_size, num_wires) if no_observable_provided else (-1, bin_size)
            )
            return [
                self._samples_to_counts(bin_sample, mp, num_wires)
                for bin_sample in samples.reshape(shape)
            ]

        return (
            samples.T.reshape((num_wires, bin_size, -1))
            if no_observable_provided
            else samples.reshape((bin_size, -1))
        )

    def expval(self, observable, shot_range=None, bin_size=None, circuit_index=None):
        """
        Overwrites the method in QubiDevice class to accomodate batch submission
        of circuits. 
        """        
        # exact expectation value
        if self.shots is None:
            try:
                eigvals = self._asarray(
                    (
                        observable.eigvals()
                        if not isinstance(observable, MeasurementValue)
                        # Indexing a MeasurementValue gives the output of the processing function
                        # for that index as a binary number.
                        else [
                            observable[i]
                            for i in range(2 ** len(observable.measurements))
                        ]
                    ),
                    dtype=self.R_DTYPE,
                )
            except qml.operation.EigvalsUndefinedError as e:
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute analytic expectations of {observable.name}."
                ) from e

            prob = self.probability(wires=observable.wires, circuit_index=circuit_index)
            # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
            return self._dot(prob, eigvals)

        # estimate the ev
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)
        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        return np.squeeze(np.mean(samples, axis=axis))

    def var(self, observable, shot_range=None, bin_size=None, circuit_index=None):
        """
        Overwrites the method in QubiDevice class to accomodate batch submission
        of circuits. 
        """
        # exact variance value
        if self.shots is None:
            try:
                eigvals = self._asarray(
                    (
                        observable.eigvals()
                        if not isinstance(observable, MeasurementValue)
                        # Indexing a MeasurementValue gives the output of the processing function
                        # for that index as a binary number.
                        else [
                            observable[i]
                            for i in range(2 ** len(observable.measurements))
                        ]
                    ),
                    dtype=self.R_DTYPE,
                )
            except qml.operation.EigvalsUndefinedError as e:
                # if observable has no info on eigenvalues, we cannot return this measurement
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute analytic variance of {observable.name}."
                ) from e

            prob = self.probability(wires=observable.wires, circuit_index=circuit_index)
            # In case of broadcasting, `prob` has two axes and these are a matrix-vector products
            return self._dot(prob, (eigvals**2)) - self._dot(prob, eigvals) ** 2

        # estimate the variance
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size, circuit_index=circuit_index)
        # With broadcasting, we want to take the variance over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        return np.squeeze(np.var(samples, axis=axis))


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

    def generate_samples(self, circuit_index=None):
        """Generates samples by random sampling with the probabilities returned by the simulator."""
        number_of_states = 2**self.num_wires
        samples = self.sample_basis_states(number_of_states, self.prob(circuit_index))
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

    def generate_samples(self, circuit_index=None):
        """Generates samples from the qpu.

        Note that the order of the samples returned here is not indicative of the order in which
        the experiments were done, but is instead controlled by a random shuffle (and hence
        set by numpy random seed).
        """
        number_of_states = 2**self.num_wires
        counts = np.rint(
            self.prob(circuit_index) * self.shots,
            out=np.zeros(number_of_states, dtype=int),
            casting="unsafe",
        )
        samples = np.repeat(np.arange(number_of_states), counts)
        np.random.shuffle(samples)
        return QubitDevice.states_to_binary(samples, self.num_wires)
