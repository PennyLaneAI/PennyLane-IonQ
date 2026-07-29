# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the to_qasm3 serializer.
"""
import pytest

import pennylane as qp
from pennylane.measurements import MidMeasureMP
from pennylane.wires import Wires

from pennylane_ionq import GPI, GPI2, MS, XX, YY, ZZ, to_qasm3
from pennylane_ionq.qasm3 import operations_to_qasm3


def _parse(qasm):
    """Validate a program against the OpenQASM 3 grammar if the reference
    parser is available."""
    openqasm3 = pytest.importorskip("openqasm3")
    openqasm3.parse(qasm)


def _mid_measure(wire, reset=False, postselect=None):
    return MidMeasureMP(Wires(wire), reset=reset, postselect=postselect)


class TestToQasm3:
    """Tests serializing circuits to OpenQASM 3.0 programs."""

    def test_basic_circuit(self):
        """A plain circuit serializes with header, registers, gates, and
        terminal measurements on all wires."""
        tape = qp.tape.QuantumScript(
            [qp.Hadamard(0), qp.CNOT(wires=[0, 1])], [qp.sample()]
        )
        qasm = to_qasm3(tape)
        expected = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "bit[2] c;",
                "h q[0];",
                "cx q[0],q[1];",
                "c[0] = measure q[0];",
                "c[1] = measure q[1];",
            ]
        )
        assert qasm == expected + "\n"
        _parse(qasm)

    def test_qnode_input(self):
        """A QNode input returns a wrapper that accepts the QNode arguments."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.sample()

        qasm = to_qasm3(circuit)(0.5)
        assert "rx(0.5) q[0];" in qasm
        _parse(qasm)

    def test_mid_measure(self):
        """Mid-circuit measurements are recorded in the mcms register."""
        tape = qp.tape.QuantumScript(
            [qp.Hadamard(0), _mid_measure(0), qp.PauliX(1)], [qp.sample()]
        )
        qasm = to_qasm3(tape)
        assert "bit[1] mcms;" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        assert "reset" not in qasm
        _parse(qasm)

    def test_mid_measure_reset(self):
        """A mid-circuit measurement with reset emits an explicit reset,
        enabling qubit reuse."""
        tape = qp.tape.QuantumScript(
            [qp.Hadamard(0), _mid_measure(0, reset=True), qp.PauliX(0)],
            [qp.sample()],
        )
        qasm = to_qasm3(tape)
        lines = qasm.splitlines()
        measure_index = lines.index("mcms[0] = measure q[0];")
        assert lines[measure_index + 1] == "reset q[0];"
        assert lines[measure_index + 2] == "x q[0];"
        _parse(qasm)

    def test_multiple_mid_measures(self):
        """Multiple mid-circuit measurements get consecutive mcms bits in
        circuit order."""
        tape = qp.tape.QuantumScript(
            [
                qp.Hadamard(0),
                _mid_measure(0, reset=True),
                qp.CNOT(wires=[0, 1]),
                _mid_measure(1),
            ],
            [qp.sample()],
        )
        qasm = to_qasm3(tape)
        assert "bit[2] mcms;" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        assert "mcms[1] = measure q[1];" in qasm
        _parse(qasm)

    def test_postselect_raises(self):
        """Postselecting mid-circuit measurements are not supported."""
        tape = qp.tape.QuantumScript(
            [qp.Hadamard(0), _mid_measure(0, postselect=0)], [qp.sample()]
        )
        with pytest.raises(NotImplementedError, match="postselection"):
            to_qasm3(tape)

    def test_conditional_raises(self):
        """Classically controlled operations (qp.cond) are out of scope."""
        with qp.tape.QuantumTape() as tape:
            qp.Hadamard(0)
            m = qp.measure(0)
            qp.cond(m, qp.PauliX)(0)
            qp.sample()
        with pytest.raises(ValueError):
            to_qasm3(tape)

    def test_unsupported_gate_raises(self):
        """A gate with no QASM spelling and no decomposition raises."""

        class NoDecompOp(qp.operation.Operation):
            """Dummy operation with no QASM spelling or decomposition."""

            num_wires = 1

        tape = qp.tape.QuantumScript([NoDecompOp(wires=0)], [qp.sample()])
        with pytest.raises(ValueError):
            to_qasm3(tape)

    def test_decomposition_fallback(self):
        """Gates outside the QASM gate set decompose to supported ones."""
        tape = qp.tape.QuantumScript([qp.IsingXX(0.5, wires=[0, 1])], [qp.sample()])
        qasm = to_qasm3(tape)
        assert "IsingXX" not in qasm
        _parse(qasm)

    def test_rotations(self):
        """Diagonalizing gates are appended when rotations=True."""
        tape = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.expval(qp.PauliX(0))])
        qasm = to_qasm3(tape)
        assert qasm.count("h q[0];") == 2
        assert to_qasm3(tape, rotations=False).count("h q[0];") == 1

    def test_measure_all_false(self):
        """measure_all=False only measures the terminally measured wires."""
        tape = qp.tape.QuantumScript(
            [qp.Hadamard(0), qp.CNOT(wires=[0, 1])], [qp.sample(wires=1)]
        )
        qasm = to_qasm3(tape, measure_all=False)
        assert "bit[1] c;" in qasm
        assert "c[0] = measure q[1];" in qasm
        assert "measure q[0];" not in qasm
        _parse(qasm)

    def test_precision(self):
        """The precision argument controls parameter formatting."""
        tape = qp.tape.QuantumScript([qp.RX(0.123456789, wires=0)], [qp.sample()])
        qasm = to_qasm3(tape, precision=3)
        assert "rx(0.123) q[0];" in qasm

    def test_global_phase(self):
        """GlobalPhase emits a gphase statement with no qubit operands and
        the opposite sign convention."""
        tape = qp.tape.QuantumScript(
            [qp.GlobalPhase(0.5), qp.PauliX(0)], [qp.sample()]
        )
        qasm = to_qasm3(tape)
        assert "gphase(-0.5);" in qasm
        _parse(qasm)

    def test_empty_circuit(self):
        """A circuit with no wires serializes to just the header."""
        tape = qp.tape.QuantumScript([], [])
        assert to_qasm3(tape) == 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'

    @pytest.mark.parametrize("gate_class", [XX, YY, ZZ])
    def test_plugin_ising_gates_decompose(self, gate_class):
        """The plugin's Ising gates serialize via their core-gate decomposition."""
        tape = qp.tape.QuantumScript(
            [gate_class(0.5, wires=[0, 1]), _mid_measure(0)], [qp.sample()]
        )
        qasm = to_qasm3(tape)
        assert gate_class.__name__ not in qasm
        assert "mcms[0] = measure q[0];" in qasm
        _parse(qasm)

    def test_native_gateset(self):
        """Native gates serialize directly, in turns, without stdgates.inc."""
        tape = qp.tape.QuantumScript(
            [
                GPI(0.5, wires=0),
                GPI2(0, wires=1),
                _mid_measure(0, reset=True),
                MS(0, 0.5, wires=[0, 1]),
            ],
            [qp.sample()],
        )
        qasm = to_qasm3(tape, gateset="native")
        assert "stdgates.inc" not in qasm
        assert "gpi(0.5) q[0];" in qasm
        assert "gpi2(0) q[1];" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        assert "reset q[0];" in qasm
        assert "ms(0,0.5,0.25) q[0],q[1];" in qasm
        _parse(qasm)

    def test_native_gateset_rejects_qis_gate(self):
        """Abstract gates are not serialized into a native-gateset program."""
        tape = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.sample()])
        with pytest.raises(ValueError):
            to_qasm3(tape, gateset="native")

    def test_operations_to_qasm3(self):
        """The operations-level serializer emits no terminal measurements."""
        ops = [qp.Hadamard(0), _mid_measure(0, reset=True), qp.PauliX(0)]
        qasm = operations_to_qasm3(ops, wires=Wires([0, 1]))
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

    def test_operations_to_qasm3_custom_wire_labels(self):
        """Wire labels map to qubit indices by their position in wires."""
        ops = [qp.Hadamard("b"), _mid_measure("a")]
        qasm = operations_to_qasm3(ops, wires=Wires(["a", "b"]))
        assert "h q[1];" in qasm
        assert "mcms[0] = measure q[0];" in qasm
        _parse(qasm)

    def test_mcm_qnode_end_to_end(self):
        """A QNode with measure-and-reuse serializes to a valid program."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.measure(0, reset=True)
            qp.CNOT(wires=[0, 1])
            return qp.sample()

        qasm = to_qasm3(circuit)()
        expected = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "bit[2] c;",
                "bit[1] mcms;",
                "h q[0];",
                "mcms[0] = measure q[0];",
                "reset q[0];",
                "cx q[0],q[1];",
                "c[0] = measure q[0];",
                "c[1] = measure q[1];",
            ]
        )
        assert qasm == expected + "\n"
        _parse(qasm)
