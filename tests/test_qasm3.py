# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Tests for the OpenQASM 3.0 serializer.
"""
import pytest

import pennylane as qp
from pennylane.measurements import MidMeasureMP
from pennylane.wires import Wires

from pennylane_ionq import GPI, GPI2, MS, XX, YY, ZZ
from pennylane_ionq.qasm3 import operations_to_qasm3


def _parse(qasm):
    """Validate a program against the OpenQASM 3 grammar if the reference
    parser is available."""
    openqasm3 = pytest.importorskip("openqasm3")
    openqasm3.parse(qasm)


class TestOperationsToQasm3:
    """Tests serializing operation lists to OpenQASM 3.0 programs."""

    def test_basic_program(self):
        """A plain operation list serializes with header, qubit register, and
        gates, without terminal measurements."""
        qasm = operations_to_qasm3([qp.Hadamard(0), qp.CNOT(wires=[0, 1])], wires=[0, 1])
        expected = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "h q[0];",
                "cx q[0], q[1];",
            ]
        )
        assert qasm == expected + "\n"
        _parse(qasm)

    def test_mid_measure(self):
        """Mid-circuit measurements are recorded in the mcms register."""
        ops = [qp.Hadamard(0), MidMeasureMP(Wires(0)), qp.PauliX(1)]
        qasm = operations_to_qasm3(ops, wires=[0, 1])
        assert "bit[1] mcms;" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        assert "reset" not in qasm
        _parse(qasm)

    def test_mid_measure_reset(self):
        """A mid-circuit measurement with reset emits an explicit reset,
        enabling qubit reuse."""
        ops = [qp.Hadamard(0), MidMeasureMP(Wires(0), reset=True), qp.PauliX(0)]
        qasm = operations_to_qasm3(ops, wires=[0, 1])
        expected = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "bit[1] mcms;",
                "h q[0];",
                "mcms[0] = measure q[0];",
                "reset q[0];",
                "x q[0];",
            ]
        )
        assert qasm == expected + "\n"
        _parse(qasm)

    def test_multiple_mid_measures(self):
        """Multiple mid-circuit measurements get consecutive mcms bits in
        circuit order."""
        ops = [
            qp.Hadamard(0),
            MidMeasureMP(Wires(0), reset=True),
            qp.CNOT(wires=[0, 1]),
            MidMeasureMP(Wires(1)),
        ]
        qasm = operations_to_qasm3(ops, wires=[0, 1])
        expected = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "bit[2] mcms;",
                "h q[0];",
                "mcms[0] = measure q[0];",
                "reset q[0];",
                "cx q[0], q[1];",
                "mcms[1] = measure q[1];",
            ]
        )
        assert qasm == expected + "\n"
        _parse(qasm)

    def test_postselect_raises(self):
        """Postselecting mid-circuit measurements are not supported."""
        ops = [qp.Hadamard(0), MidMeasureMP(Wires(0), postselect=0)]
        with pytest.raises(NotImplementedError, match="postselection"):
            operations_to_qasm3(ops, wires=[0])

    def test_conditional_raises(self):
        """Classically controlled operations (qp.cond) are out of scope."""
        with qp.tape.QuantumTape() as tape:
            qp.Hadamard(0)
            m = qp.measure(0)
            qp.cond(m, qp.PauliX)(0)
        with pytest.raises(ValueError):
            operations_to_qasm3(tape.operations, wires=[0])

    def test_unsupported_gate_raises(self):
        """A gate with no QASM spelling and no decomposition raises."""

        class NoDecompOp(qp.operation.Operation):
            """Dummy operation with no QASM spelling or decomposition."""

            num_wires = 1

        with pytest.raises(ValueError):
            operations_to_qasm3([NoDecompOp(wires=0)], wires=[0])

    def test_decomposition_fallback(self):
        """Gates outside the QASM gate set decompose to supported ones."""
        qasm = operations_to_qasm3([qp.IsingXX(0.5, wires=[0, 1])], wires=[0, 1])
        assert "IsingXX" not in qasm
        _parse(qasm)

    def test_precision(self):
        """The precision argument controls parameter formatting."""
        qasm = operations_to_qasm3([qp.RX(0.123456789, wires=0)], wires=[0], precision=3)
        assert "rx(0.123) q[0];" in qasm

    def test_global_phase(self):
        """GlobalPhase emits a gphase statement with no qubit operands and
        the opposite sign convention."""
        qasm = operations_to_qasm3([qp.GlobalPhase(0.5), qp.PauliX(0)], wires=[0])
        assert "gphase(-0.5);" in qasm
        _parse(qasm)

    @pytest.mark.parametrize("gate_class", [XX, YY, ZZ])
    def test_plugin_ising_gates_decompose(self, gate_class):
        """The plugin's Ising gates serialize via their core-gate decomposition."""
        ops = [gate_class(0.5, wires=[0, 1]), MidMeasureMP(Wires(0))]
        qasm = operations_to_qasm3(ops, wires=[0, 1])
        assert gate_class.__name__ not in qasm
        assert "mcms[0] = measure q[0];" in qasm
        _parse(qasm)

    def test_native_gateset(self):
        """Native gates serialize directly, in turns, without stdgates.inc."""
        ops = [
            GPI(0.5, wires=0),
            GPI2(0, wires=1),
            MidMeasureMP(Wires(0), reset=True),
            MS(0, 0.5, wires=[0, 1]),
        ]
        qasm = operations_to_qasm3(ops, wires=[0, 1], gateset="native")
        assert "stdgates.inc" not in qasm
        assert "gpi(0.5) q[0];" in qasm
        assert "gpi2(0) q[1];" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        assert "reset q[0];" in qasm
        assert "ms(0,0.5,0.25) q[0], q[1];" in qasm
        _parse(qasm)

    def test_native_gateset_rejects_qis_gate(self):
        """Abstract gates are not serialized into a native-gateset program."""
        with pytest.raises(ValueError):
            operations_to_qasm3([qp.Hadamard(0)], wires=[0], gateset="native")

    def test_custom_wire_labels(self):
        """Wire labels map to qubit indices by their position in wires."""
        ops = [qp.Hadamard("b"), MidMeasureMP(Wires("a"))]
        qasm = operations_to_qasm3(ops, wires=["a", "b"])
        assert "h q[1];" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        _parse(qasm)
