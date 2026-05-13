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
"""Tests for shotwise output in IonQ jobs."""

import numpy as np
import pennylane as qml


class TestShotwiseOutput:
    """Tests for shotwise output is retrieved correctly in IonQ jobs."""

    @staticmethod
    def _assert_most_frequent_sample_matches(samples, expected_sample):
        """Assert the most frequent observed sample matches the expected sample."""
        unique_samples, counts = np.unique(samples, axis=0, return_counts=True)
        most_frequent_sample = unique_samples[np.argmax(counts)]
        np.testing.assert_array_equal(most_frequent_sample, expected_sample)

    def test_shotwise_output_single_circuit(self, requires_api):
        """When shotwise is enabled, shotwise output is retrieved
        on a single circuit job.
        """

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model="forte-enterprise-1",
            shotwise=True,
        )

        with qml.tape.QuantumTape(shots=100) as tape:
            qml.X(wires=["q1"])
            qml.sample(wires=["q0", "q1", "q2"])

        results = dev.batch_execute([tape])

        self._assert_most_frequent_sample_matches(results[0], np.array([0, 1, 0]))

    def test_shotwise_output_two_circuits(self, requires_api):
        """When shotwise is enabled, shotwise output is retrieved
        on a two circuit job. The shots results are reversed for
        tape2 when compared to tape1, verifying that endianess is 
        handled correctly.
        """

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model="forte-enterprise-1",
            shotwise=True,
        )

        with qml.tape.QuantumTape(shots=100) as tape1:
            qml.X(wires=["q0"])
            qml.sample(wires=["q0", "q1", "q2"])

        with qml.tape.QuantumTape(shots=100) as tape2:
            qml.X(wires=["q2"])
            qml.sample(wires=["q0", "q1", "q2"])

        results = dev.batch_execute([tape1, tape2])

        self._assert_most_frequent_sample_matches(results[0], np.array([1, 0, 0]))
        self._assert_most_frequent_sample_matches(results[1], np.array([0, 0, 1]))

    def test_shotwise_output_disabled(self, requires_api):
        """When shotwise is disabled, shotwise output is generated locally."""

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model="aria-1",
            shotwise=False,
        )

        @qml.qnode(dev, shots=100)
        def circuit():
            qml.X(wires="q0")
            return (qml.sample(wires=["q0", "q1", "q2"]),)

        samples = circuit()
        self._assert_most_frequent_sample_matches(samples[0], np.array([1, 0, 0]))

    def test_shotwise_output_noise_is_none(self, requires_api):
        """When noise_model is not set, shotwise output is generated locally."""

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model=None,
            shotwise=True,
        )

        @qml.qnode(dev, shots=100)
        def circuit():
            qml.X(wires="q2")
            return (qml.sample(wires=["q0", "q1", "q2"]),)

        samples = circuit()
        self._assert_most_frequent_sample_matches(samples[0], np.array([0, 0, 1]))
