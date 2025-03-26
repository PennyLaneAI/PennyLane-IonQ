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
"""Tests for circuits that use the 'pauliexp' IonQ gate PennyLane"""

import numpy as np
import pennylane as qml
from scipy.sparse import csr_matrix


class TestIonQPauliexp:
    """Tests for circuits that use the 'pauliexp' IonQ gate PennyLane."""

    def test_evolution_object_created_from_hamiltonian_1(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        coeffs = [0.1, 0.2, 0.3]
        ops = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliX(0)]

        H = qml.Hamiltonian(coeffs, ops)

        time = 0.4
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])
        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results are not equal."

    def test_evolution_object_created_from_hamiltonian_2(self, requires_api):

        dev = qml.device("ionq.simulator", wires=3, gateset="qis")

        coeffs = [1.0, -2.0, 3.0]
        ops = [
            qml.PauliX(0) @ qml.Identity(1) @ qml.PauliZ(2),
            qml.PauliX(0) @ qml.Y(1) @ qml.Identity(2),
            qml.Identity(0) @ qml.Y(1) @ qml.PauliZ(2),
        ]

        H = qml.Hamiltonian(coeffs, ops)

        time = 1
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1, 2])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=3)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results are not equal."

    def test_evolution_object_created_from_sparse_hamiltonian_1(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        sparse_matrix = csr_matrix(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )

        dense_matrix = sparse_matrix.toarray()
        hermitian_matrix = (dense_matrix + dense_matrix.T.conj()) / 2
        hermitian_sparse = csr_matrix(hermitian_matrix)

        H_sparse = qml.SparseHamiltonian(hermitian_sparse, wires=[0, 1])

        time = 2
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H_sparse, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results are not equal."

    def test_evolution_object_created_from_sparse_hamiltonian_2(self, requires_api):

        dev = qml.device("ionq.simulator", wires=3, gateset="qis")

        sparse_matrix = csr_matrix(
            [
                [1.0, 2.0, 0.0, 1.0],
                [0.0, -1.0, 3.0, 0.0],
                [0.0, 1.0, -1.0, 1.0],
                [-1.0, -2.0, 0.0, 7.0],
            ]
        )

        dense_matrix = sparse_matrix.toarray()
        hermitian_matrix = (dense_matrix + dense_matrix.T.conj()) / 2
        hermitian_sparse = csr_matrix(hermitian_matrix)

        H_sparse = qml.SparseHamiltonian(hermitian_sparse, wires=[0, 1])

        time = 3
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H_sparse, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results are not equal."
