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
from pennylane.ops.op_math import Prod, SProd, Exp
from pennylane.tape import QuantumScript

from pennylane_ionq.exceptions import (
    ComplexEvolutionCoefficientsNotSupported,
    NotSupportedEvolutionInstance,
    OperatorNotSupportedInEvolutionGateGenerator,
)


class TestIonQPauliexp:
    """Tests for circuits that use the 'pauliexp' IonQ gate PennyLane."""

    def test_instance_of_evolution_gate_not_supported(self):
        """Test that a relevant exception is raised when the current
        instance Evolution gate is not supported.
        """

        dev = qml.device("ionq.simulator", wires=1, gateset="qis")

        op = qml.S(0)

        time = 1
        tape = qml.tape.QuantumScript([qml.evolve(op, time)], [qml.probs(wires=[0])])

        with pytest.raises(
            NotSupportedEvolutionInstance,
            match="The current instance of the Evolution gate is not supported.",
        ):
            dev.batch_execute([tape])

    def test_operator_in_generator_of_evolution_gate_not_supported(self):
        """Test a relevant exception is raised when the generator
        of the Evolution gate is not supported.
        """

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.sum(qml.H(1), qml.PauliZ(0))

        time = 1
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        with pytest.raises(
            OperatorNotSupportedInEvolutionGateGenerator,
            match=re.escape("Unsupported operator in generator of Evolution gate: H(1)"),
        ):
            dev.batch_execute([tape])

    def test_operand_not_supported_for_evolution_gate(self):
        """Test a relevant exception is raised when triyng to use
        an Evolution gate generated via an unsupported operand.
        """

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.H(0) @ qml.PauliX(1)

        time = 1
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        with pytest.raises(
            ValueError,
            match="Operand Hadamard is not supported for Evolution gate. Supported operands: PauliX, PauliY, PauliZ, Identity.",
        ):
            dev.batch_execute([tape])

    def test_complex_evolution_operators_not_supported(self):
        """Test an exception is thrown when coefficients for Evolution
        gate are complex since IonQ API does not support this."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = SProd(1j, qml.Hamiltonian([1.0], [qml.PauliX(0)]))

        time = 1.2
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        with pytest.raises(
            ComplexEvolutionCoefficientsNotSupported,
            match="Complex coefficients in Evolution gates are not supported.",
        ):
            dev.batch_execute([tape])

    def test_identity_evolution_gate_generator(self, requires_api):
        """Test the implementation of Evolution gate using pauliexp gate
        works when applied Evolution is generated from an Identity operator.
        """

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = 3 * qml.Identity(0) @ qml.Identity(1)

        time = 1.5
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(result_ionq, result_simulator, atol=1e-2)

    @pytest.mark.parametrize(
        "wires, coeffs, ops",
        [
            (
                [0, 1],
                [0.1, 0.2, 0.3, 0.4],
                [
                    qml.Identity(0) @ qml.Identity(1),
                    qml.PauliX(0) @ qml.PauliX(1),
                    qml.PauliX(1),
                    qml.PauliX(0),
                ],
            ),
            (
                [0, 1, 2],
                [1.0, -2.0, 3.0],
                [
                    qml.PauliX(0) @ qml.Identity(1) @ qml.PauliZ(2),
                    qml.PauliX(0) @ qml.Y(1) @ qml.Identity(2),
                    qml.Identity(0) @ qml.Y(1) @ qml.PauliZ(2),
                ],
            ),
        ],
        ids=lambda val: f"{val}",
    )
    def test_evolution_object_created_from_hamiltonian(self, wires, coeffs, ops, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via a Hamiltonian term works.
        """

        no_wires = len(wires)
        dev = qml.device("ionq.simulator", wires=no_wires, gateset="qis")
        H = 2 * qml.Hamiltonian(coeffs, ops)

        time = 7
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=wires)])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=no_wires)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sparse_hamiltonian_1(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via a sparse Hamiltonian works.
        """

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
        tape = qml.tape.QuantumScript([qml.evolve(H_sparse, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sparse_hamiltonian_2(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via a sparse Hamiltonian works.
        """

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

        H_sparse = qml.SparseHamiltonian(hermitian_sparse, wires=[0, 1])

        time = 3
        tape = qml.tape.QuantumScript([qml.evolve(H_sparse, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        # Pennylane simulator returns incorrect
        # results for this test, probably because Pauli
        # strings in which the operator is decomposed
        # do not commute.

        results_qiskit_statevector_sim = [
            0.90082792,
            0.07257902,
            0.01235616,
            0.01423689,
        ]

        assert np.allclose(
            result_ionq, results_qiskit_statevector_sim, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_1(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via an SProd term works.
        """

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H1 = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        H2 = qml.Hamiltonian([-0.5], [qml.PauliZ(1)])
        sum_H = 2 * SProd(2.5, H1) + SProd(3.1, H2)

        time = 3
        tape = qml.tape.QuantumScript([qml.evolve(sum_H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sprod_2(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via an SProd term works.
        """

        dev = qml.device("ionq.simulator", wires=1, gateset="qis")

        H = SProd(2.5, qml.PauliX(0))
        tape = qml.tape.QuantumScript([qml.evolve(H)], [qml.probs(wires=[0])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_via_exp(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from a Hamiltonian constructed via an Exp term works."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = qml.PauliX(0) + 0.5 * qml.PauliY(1)
        t = 1
        U = Exp(t * H)
        sprod_op = U.base

        time = 3
        tape = qml.tape.QuantumScript([qml.evolve(sprod_op, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_prod(self, requires_api):
        """Test that the implementation of Evolution gate derived
        from an Hamiltonian constructed via Prod term works."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = 2 * Prod(qml.X(0), qml.PauliZ(1))

        time = 2
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_sum(self, requires_api):
        """Test that the implementation of Evolution gate
        derived from an Hamiltonian constructed via sum works."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = 3 * qml.sum(qml.PauliX(0), qml.PauliZ(1))

        time = 2
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_prod(self, requires_api):
        """Test that the implementation of Evolution gate
        derived from an Hamiltonian constructed via prod works."""

        dev = qml.device("ionq.simulator", wires=3, gateset="qis")

        H = 1.5 * qml.prod(qml.PauliX(0), qml.PauliZ(1), qml.PauliY(2))

        time = 3
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=3)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_hermitian_matrix_1(self, requires_api):
        """Test that the implementation of Evolution gate
        derived from a Hamiltonian constructed via a Hermitian matrix."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H_matrix = np.array(
            [[1, 1 + 1j, 0, -1j], [1 - 1j, 3, 2, 0], [0, 2, 0, 1j], [1j, 0, -1j, 1]]
        )

        hermitian_op = qml.Hermitian(H_matrix, wires=[0, 1])

        H = qml.Hamiltonian([2.0], [hermitian_op])

        time = 7
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        # Pennylane simulator returns incorrect
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
        """Test that the implementation of Evolution gate
        derived from a Hamiltonian constructed via a Hermitian matrix."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H_matrix = np.array([[1, 0, 0, 0], [0, 0.5, 0.3, 0], [0, 0.3, 0.5, 0], [0, 0, 0, 1]])

        hermitian_op = qml.Hermitian(H_matrix, wires=[0, 1])

        H = qml.Hamiltonian([2.0], [hermitian_op])

        time = 7
        tape = qml.tape.QuantumScript([qml.evolve(H, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."

    def test_evolution_object_created_from_exp_operator(self, requires_api):
        """Test that the implementation of Evolution gate
        derived from a Hamiltonian constructed via an Exp matrix."""

        dev = qml.device("ionq.simulator", wires=2, gateset="qis")

        H = 2.5 * qml.PauliX(0) + 0.5 * qml.PauliY(1)
        t = 3
        U = Exp(t * H)

        time = 2
        tape = qml.tape.QuantumScript([qml.evolve(U, time)], [qml.probs(wires=[0, 1])])

        result_ionq = dev.batch_execute([tape])

        simulator = qml.device("default.qubit", wires=2)
        result_simulator = qml.execute([tape], simulator)

        assert np.allclose(
            result_ionq, result_simulator, atol=1e-2
        ), "The IonQ and simulator results do not agree."
