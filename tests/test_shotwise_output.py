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

    def test_shotwise_output(self, requires_api):
        """When shotwise is enabled, shotwise output is retrieved.
        The shots results are reversed for tape2 when compared to
        tape1, ensuring endianess is handled correctly.
        """

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model="aria-1",
            shotwise=True,
        )

        with qml.tape.QuantumTape(shots=3) as tape1:
            qml.X(wires=["q0"])
            qml.sample(wires=["q0", "q1", "q2"])

        with qml.tape.QuantumTape(shots=3) as tape2:
            qml.X(wires=["q2"])
            qml.sample(wires=["q0", "q1", "q2"])

        results = dev.batch_execute([tape1, tape2])

        np.testing.assert_array_equal(results[0], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])

        np.testing.assert_array_equal(results[1], [[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    def test_shotwise_output_disabled(self, requires_api):
        """When shotwise is disabled, shotwise output is generated locally."""

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model="aria-1",
            shotwise=False,
        )

        @qml.qnode(dev, shots=3)
        def circuit():
            qml.X(wires="q0")
            return (qml.sample(wires=["q0", "q1", "q2"]),)

        samples = circuit()
        np.testing.assert_array_equal(samples[0], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])

    def test_shotwise_output_noise_is_none(self, requires_api):
        """When noise_mode is not set, shotwise output is generated locally."""

        dev = qml.device(
            "ionq.simulator",
            wires=["q0", "q1", "q2"],
            gateset="qis",
            noise_model=None,
            shotwise=True,
        )

        @qml.qnode(dev, shots=3)
        def circuit():
            qml.X(wires="q2")
            return (qml.sample(wires=["q0", "q1", "q2"]),)

        samples = circuit()
        np.testing.assert_array_equal(samples[0], [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
