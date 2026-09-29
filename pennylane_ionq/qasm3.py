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
Serialization of PennyLane operations to OpenQASM 3.0 programs for
``ionq.qasm3.v1`` jobs, supporting mid-circuit measurements and qubit reuse.

Adapted from ``pennylane.io.to_openqasm`` (PennyLane, Apache License 2.0).
"""

from pennylane.devices.preprocess import decompose
from pennylane.ops import MidMeasure
from pennylane.tape import QuantumScript
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.wires import Wires, WiresLike

OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "SX": "sx",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
    "GlobalPhase": "gphase",
}
"""dict[str, str]: Maps PennyLane gate names to OpenQASM 3 gate names.

``gphase`` is a QASM 3 builtin; all other gates are defined in stdgates.inc,
which keeps the ``u1``/``u2``/``u3`` spellings for backwards compatibility.
"""

NATIVE_QASM3_GATES = {
    "GPI": "gpi",
    "GPI2": "gpi2",
    "MS": "ms",
}
"""dict[str, str]: Maps IonQ native gates to their OpenQASM 3 spellings.

IonQ accepts these directly in OpenQASM 3.0 programs, without an include;
their parameters are specified in turns, matching the plugin's operations.
"""

_GATESET_QASM3_GATES = {
    "qis": OPENQASM_GATES,
    "native": NATIVE_QASM3_GATES,
}


def _param_string(parameters, precision):
    if precision is not None:
        return ",".join(f"{p:.{precision}}" for p in parameters)
    return ",".join(str(p) for p in parameters)


def _gate_string(op, wires: Wires, precision: None | int, gates: dict) -> str:
    try:
        gate = gates[op.name]
    except KeyError as e:
        raise ValueError(
            f"Operation {op.name} not supported by the PennyLane-IonQ QASM 3 serializer"
        ) from e

    if op.name == "GlobalPhase":
        # QASM 3's gphase takes no qubit operands, and its convention
        # (exp(i*gamma)) is opposite in sign to GlobalPhase (exp(-i*phi)).
        return f"gphase({_param_string([-p for p in op.parameters], precision)});"

    wire_labels = ", ".join(f"q[{wires.index(w)}]" for w in op.wires.tolist())
    params = f"({_param_string(op.parameters, precision)})" if op.num_params > 0 else ""

    return f"{gate}{params} {wire_labels};"


def _mid_measure_string(op: MidMeasure, wires: Wires, mcm_index: int) -> str:
    if op.postselect is not None:
        raise NotImplementedError(
            f"Unable to translate mid circuit measurement with postselection {op}"
        )

    wire = f"q[{wires.index(op.wires[0])}]"
    line = f"mcms[{mcm_index}] = measure {wire};"

    if op.reset:
        # QASM 3 has an explicit reset statement; this is what enables qubit
        # reuse, returning the measured qubit to |0> for later operations.
        line += f"\nreset {wire};"

    return line


def operations_to_qasm3(operations, wires: WiresLike, gateset: str = "qis", precision=None) -> str:
    """Serializes a list of operations to an OpenQASM 3.0 program without
    terminal measurements.

    This is the serializer behind the ``ionq.qasm3.v1`` device submission
    path; terminal measurement handling is left to the backend. Operations
    without a QASM spelling are decomposed into the target gateset.

    Args:
        operations (Iterable[Operation]): the operations to serialize, which
            may include mid-circuit measurements and resets
        wires (WiresLike): all device wires; wire labels are mapped to qubit
            indices by their position
        gateset (str): the target gateset, either ``"qis"`` (stdgates.inc
            spellings) or ``"native"`` (IonQ native gates). Defaults to ``qis``.
        precision (int or None): number of decimal digits to display for the
            parameters

    Returns:
        str: OpenQASM 3.0 program corresponding to the operations

    Raises:
        ValueError: if the operations contain a classically controlled
            operation (``qp.cond``), or an operation that cannot be
            decomposed into the target gateset
        NotImplementedError: if the operations contain a mid-circuit
            measurement with postselection

    **Example**

    .. code-block:: python

        with qp.queuing.AnnotatedQueue() as q:
            qp.Hadamard(0)
            qp.measure(0, reset=True)
            qp.CNOT(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)

    >>> print(operations_to_qasm3(tape.operations, wires=[0, 1]))
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[1] mcms;
    h q[0];
    mcms[0] = measure q[0];
    reset q[0];
    cx q[0], q[1];
    """
    wires = Wires(wires)
    gates = _GATESET_QASM3_GATES[gateset]

    operations = [op for op in operations if op.name not in ("Barrier", "Snapshot")]
    [tape], _ = convert_to_numpy_parameters(QuantumScript(operations))

    def stopping_condition(op):
        return op.name in gates or isinstance(op, MidMeasure)

    [tape], _ = decompose(
        tape,
        target_gates=gates.keys() | {"MidMeasure"},
        stopping_condition=stopping_condition,
        skip_initial_state_prep=False,
        name="operations_to_qasm3",
        error=ValueError,
    )

    lines = ["OPENQASM 3.0;"]
    if gateset != "native":
        # IonQ native gates are accepted without an include;
        # for other gatesets use stdgates.inc defs.
        lines.append('include "stdgates.inc";')
    lines.append(f"qubit[{len(wires)}] q;")

    num_mcms = sum(isinstance(op, MidMeasure) for op in tape.operations)
    if num_mcms:
        lines.append(f"bit[{num_mcms}] mcms;")

    mcm_index = 0
    for op in tape.operations:
        if isinstance(op, MidMeasure):
            lines.append(_mid_measure_string(op, wires, mcm_index))
            mcm_index += 1
        else:
            lines.append(_gate_string(op, wires, precision, gates))

    return "\n".join(lines) + "\n"
