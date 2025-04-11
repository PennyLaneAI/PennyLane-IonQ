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
import pytest
import re

from scipy.sparse import csr_matrix
from pennylane.ops.op_math import Sum, Prod, SProd, Exp

from pennylane_ionq.exceptions import (
    ComplexEvolutionCoefficientsNotSupported,
    NotSupportedEvolutionInstance,
    OperatorNotSupportedInEvolutionGateGenerator,
)


class TestIonQPauliexp:
    """Tests for circuits that use the 'pauliexp' IonQ gate PennyLane."""

    def test_instance_of_evolution_gate_not_supported(self):

        dev = qml.device("ionq.simulator", wires=1, gateset="qis")

        op = qml.S(0)

        time = 1
        with qml.tape.QuantumTape() as tape:
            qml.evolve(op, time, num_steps=1).queue()
            qml.probs(wires=[0])

        with pytest.raises(
            NotSupportedEvolutionInstance,
            match="The current instance of Evolution gate is not supported.",
        ):
            dev.batch_execute([tape])

    def test_operator_in_generator_of_evolution_gate_not_supported(self):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.sum(qml.H(1), qml.PauliZ(0))

        time = 1
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        with pytest.raises(
            OperatorNotSupportedInEvolutionGateGenerator,
            match=re.escape(
                "Unsupported operator in generator of Evolution gate: H(1)"
            ),
        ):
            dev.batch_execute([tape])

    def test_operand_not_supported_for_evolution_gate(self):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.H(0) @ qml.PauliX(1)

        time = 1
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        with pytest.raises(
            ValueError,
            match="Operand Hadamard is not supported for Evolution gate. Supported operands: PauliX, PauliY, PauliZ, Identity.",
        ):
            dev.batch_execute([tape])

    def test_complex_evolution_operators_not_supported(self):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = SProd(1j, qml.Hamiltonian([1.0], [qml.PauliX(0)]))

        time = 1.2
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        with pytest.raises(
            ComplexEvolutionCoefficientsNotSupported,
            match="Complex coefficients in Evolution gate are not supported.",
        ):
            dev.batch_execute([tape])

    def test_identity_evolutiin_gate_generator(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.Identity(0) @ qml.Identity(1)

        time = 1.5
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(result_ionq, result_simulator, atol=1e-2)

    def test_evolution_object_created_from_hamiltonian_1(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        coeffs = [0.1, 0.2, 0.3, 0.4]
        ops = [
            qml.Identity(0) @ qml.Identity(1),
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliX(1),
            qml.PauliX(0),
        ]

        H = qml.Hamiltonian(coeffs, ops)

        time = 7
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

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
        ), "The IonQ and simulator results do not agree."

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
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sparse_hamiltonian_2(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

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

        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(hermitian_matrix)

        H_sparse = qml.SparseHamiltonian(hermitian_sparse, wires=[0, 1])

        time = 3
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H_sparse, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        # TODO
        # Pennylane simulator seems to return incorrect
        # results for this test, probably because Pauli
        # strings in which the operator is decomposed
        # do not commute. 

        results_qiskit_statevector_sim = [0.06791753, 0.58025692, 0.30475063, 0.04707491]

        assert np.allclose(
            result_ionq, results_qiskit_statevector_sim, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_1(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H1 = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        H2 = qml.Hamiltonian([-0.5], [qml.PauliZ(1)])
        sum_H = SProd(2.5, H1) + SProd(3.1, H2)

        time = 3
        with qml.tape.QuantumTape() as tape:
            qml.evolve(sum_H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_2(self, requires_api):

        dev = qml.device("ionq.simulator", wires=1, gateset="qis")

        # TODO: one qubit pauliexp gates do not seem to work at IonQ

        # H = SProd(2.5, qml.PauliX(0))
        # with qml.tape.QuantumTape() as tape:
        #     qml.evolve(H).queue()
        #     qml.probs(wires=[0])
        # dev.batch_execute([tape])

        # result_ionq = dev.batch_execute([tape])

        # simulator = qml.device("default.qubit", wires=2)
        # result_simulator = qml.execute([tape], simulator)

        # assert np.allclose(
        #     result_ionq, result_simulator, atol=1e-2
        # ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_3(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.PauliX(0) + 0.5 * qml.PauliY(1)
        t = 1
        U = Exp(t * H)
        sprod_op = U.base

        time = 3
        with qml.tape.QuantumTape() as tape:
            qml.evolve(sprod_op, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_4(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = Prod(qml.X(0), qml.PauliZ(1))

        time = 2
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sum(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.sum(qml.PauliX(0), qml.PauliZ(1))

        time = 2
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_prod(self, requires_api):

        dev = qml.device("ionq.simulator", wires=3, gateset="qis")

        H = qml.prod(qml.PauliX(0), qml.PauliZ(1), qml.PauliY(2))

        time = 3
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=3)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_hermitian_matrix_1(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H_matrix = np.array(
            [[1, 1 + 1j, 0, -1j], [1 - 1j, 3, 2, 0], [0, 2, 0, 1j], [1j, 0, -1j, 1]]
        )

        hermitian_op = qml.Hermitian(H_matrix, wires=[0, 1])

        H = qml.Hamiltonian([2.0], [hermitian_op])

        time = 7
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        # Pennylane simulator seems to return incorrect
        # results for this test, probably because Pauli
        # strings in which the operator is decomposed
        # do not commute.
        results_qiskit_statevector_sim = [
            0.10784311,
            0.45583129,
            0.09056136,
            0.34576424,
        ]

        assert np.allclose(
            result_ionq, results_qiskit_statevector_sim, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_hermitian_matrix_2(self, requires_api):

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H_matrix = np.array(
            [[1, 0, 0, 0], [0, 0.5, 0.3, 0], [0, 0.3, 0.5, 0], [0, 0, 0, 1]]
        )

        hermitian_op = qml.Hermitian(H_matrix, wires=[0, 1])

        H = qml.Hamiltonian([2.0], [hermitian_op])

        time = 7
        with qml.tape.QuantumTape() as tape:
            qml.evolve(H, time, num_steps=1).queue()
            qml.probs(wires=[0, 1])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."
