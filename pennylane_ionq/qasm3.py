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
Conversion of a circuit to an OpenQASM 3.0 program.

Adapted from ``pennylane.io.to_openqasm`` (PennyLane, Apache License 2.0):
https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/io/to_openqasm.py

Changes from the original:

- Emits OpenQASM 3.0 syntax instead of OpenQASM 2.0 (``qubit[]``/``bit[]``
  declarations, assignment-form measurement, explicit ``reset``).
- Mid-circuit measurements with ``reset=True`` are supported (qubit reuse)
  instead of raising ``NotImplementedError``.
- Supports the IonQ native gateset (``gpi``/``gpi2``/``ms``), whose gates
  IonQ accepts directly in OpenQASM 3.0 programs with parameters in turns.
- Classically controlled operations (``qml.cond``) are intentionally not
  supported; this exporter is scoped to mid-circuit measurement and qubit
  reuse only. Circuits containing a ``Conditional`` raise ``ValueError``.
"""

from functools import singledispatch, wraps

from pennylane.devices.preprocess import decompose
from pennylane.operation import Operator
from pennylane.ops import MidMeasure
from pennylane.tape import QuantumScript
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.wires import Wires, WiresLike
from pennylane.workflow import construct_tape

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


# pylint: disable=unused-argument
@singledispatch
def _obj_string(
    op: Operator, wires: Wires, bit_map: dict, precision: None | int, gates: dict
) -> str:
    try:
        gate = gates[op.name]
    except KeyError as e:
        raise ValueError(f"Operation {op.name} not supported by the QASM 3 serializer") from e

    if op.name == "GlobalPhase":
        # QASM 3's gphase takes no qubit operands, and its convention
        # (exp(i*gamma)) is opposite in sign to GlobalPhase (exp(-i*phi)).
        return f"gphase({_param_string([-p for p in op.parameters], precision)});"

    wire_labels = ",".join(f"q[{wires.index(w)}]" for w in op.wires.tolist())
    params = f"({_param_string(op.parameters, precision)})" if op.num_params > 0 else ""

    return f"{gate}{params} {wire_labels};"


@_obj_string.register
def _mid_measure_str(
    op: MidMeasure, wires: Wires, bit_map: dict, precision: None | int, gates: dict
) -> str:
    if op.postselect is not None:
        raise NotImplementedError(
            f"Unable to translate mid circuit measurement with postselection {op}"
        )

    wire = f"q[{wires.index(op.wires[0])}]"
    mcm_ind = len(bit_map)
    bit_map[op] = mcm_ind
    line = f"mcms[{mcm_ind}] = measure {wire};"

    if op.reset:
        # QASM 3 has an explicit reset statement; this is what enables qubit
        # reuse, returning the measured qubit to |0> for later operations.
        line += f"\nreset {wire};"

    return line


def _program_header(gateset: str) -> list[str]:
    lines = ["OPENQASM 3.0;"]
    if gateset != "native":
        # IonQ native gates are accepted without an include; the abstract
        # gates are the stdgates.inc spellings.
        lines.append('include "stdgates.inc";')
    return lines


def _operations_qasm3_lines(operations, wires: Wires, precision: None | int, gates: dict):
    """Serializes a list of operations to OpenQASM 3.0 statement lines,
    decomposing operations without a QASM spelling into the target gate set."""
    operations = [op for op in operations if op.name not in ("Barrier", "Snapshot")]
    [transformed_tape], _ = convert_to_numpy_parameters(QuantumScript(operations))

    def stopping_condition(op):
        return op.name in gates or isinstance(op, MidMeasure)

    [new_tape], _ = decompose(
        transformed_tape,
        target_gates=gates.keys() | {"MidMeasure"},
        stopping_condition=stopping_condition,
        skip_initial_state_prep=False,
        name="to_qasm3",
        error=ValueError,
    )

    bit_map = {}
    return [_obj_string(op, wires, bit_map, precision, gates) for op in new_tape.operations]


def operations_to_qasm3(operations, wires: WiresLike, gateset: str = "qis", precision=None) -> str:
    """Serializes a list of operations to an OpenQASM 3.0 program without
    terminal measurements.

    This is the serializer behind the ``ionq.qasm3.v1`` device submission
    path; terminal measurement handling is left to the backend.

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
    """
    wires = Wires(wires)
    operations = list(operations)
    gates = _GATESET_QASM3_GATES[gateset]

    lines = _program_header(gateset)
    lines.append(f"qubit[{len(wires)}] q;")

    num_mcms = sum(isinstance(op, MidMeasure) for op in operations)
    if num_mcms:
        lines.append(f"bit[{num_mcms}] mcms;")

    lines += _operations_qasm3_lines(operations, wires, precision, gates)
    return "\n".join(lines) + "\n"


def _tape_qasm3(
    tape: QuantumScript,
    wires: Wires,
    rotations: bool,
    measure_all: bool,
    precision: None | int,
    gateset: str,
) -> str:
    """Helper function to serialize a tape as an OpenQASM 3.0 program."""
    wires = wires or tape.wires
    gates = _GATESET_QASM3_GATES[gateset]

    lines = _program_header(gateset)

    if tape.num_wires == 0:
        # empty circuit
        return "\n".join(lines) + "\n"

    # create the quantum and classical registers
    lines.append(f"qubit[{len(wires)}] q;")

    terminally_measured_wires = (
        wires
        if measure_all
        else Wires.all_wires([m.wires for m in tape.measurements if m.mv is None])
    )
    if terminally_measured_wires:
        lines.append(f"bit[{len(terminally_measured_wires)}] c;")

    num_mcms = sum(isinstance(o, MidMeasure) for o in tape.operations)
    if num_mcms:
        lines.append(f"bit[{num_mcms}] mcms;")

    operations = list(tape.operations)
    if rotations:
        # if requested, append diagonalizing gates corresponding
        # to circuit observables
        operations += tape.diagonalizing_gates

    lines += _operations_qasm3_lines(operations, wires, precision, gates)

    # apply computational basis measurements to each quantum register
    if measure_all:
        for wire in range(len(wires)):
            lines.append(f"c[{wire}] = measure q[{wire}];")
    else:
        for creg_indx, w in enumerate(terminally_measured_wires):
            qreg_indx = tape.wires.index(w)
            lines.append(f"c[{creg_indx}] = measure q[{qreg_indx}];")

    return "\n".join(lines) + "\n"


def to_qasm3(
    circuit,
    wires: WiresLike | None = None,
    rotations: bool = True,
    measure_all: bool = True,
    precision: None | int = None,
    gateset: str = "qis",
):
    """Convert a circuit to an OpenQASM 3.0 program.

    Supports mid-circuit measurements, including measurements with
    ``reset=True`` (qubit reuse). Terminal measurements are performed on all
    qubits in the computational basis by default; restrict them to the wires
    measured in the circuit with ``measure_all=False``.

    Circuits with classically controlled operations (``qml.cond``) or
    postselecting mid-circuit measurements are not supported and raise an
    error.

    Args:
        circuit (QNode or QuantumScript): the quantum circuit to be serialized.
        wires (Wires or None): the wires to use when serializing the circuit.
            Default is ``None``, such that all the wires of the circuit are
            used for serialization.
        rotations (bool): if ``True``, add gates that rotate the quantum state
            into the eigenbasis of the circuit's observables. Default is ``True``.
        measure_all (bool): if ``True``, add a computational basis measurement
            on all the qubits. Default is ``True``.
        precision (int or None): number of decimal digits to display for the
            parameters.
        gateset (str): the target gateset, either ``"qis"`` (stdgates.inc
            spellings) or ``"native"`` (IonQ native gates ``gpi``/``gpi2``/``ms``,
            with parameters in turns). Defaults to ``qis``.

    Returns:
        str or callable: If a tape is provided, the OpenQASM 3.0 program is
        returned directly. If a QNode is provided, a wrapper is returned that
        accepts the QNode's arguments and returns the program.

    **Example**

    .. code-block:: python

        dev = qml.device("ionq.simulator", wires=2, shots=1024)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.measure(0, reset=True)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

    >>> print(to_qasm3(circuit)())
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    bit[1] mcms;
    h q[0];
    mcms[0] = measure q[0];
    reset q[0];
    cx q[0],q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
    if isinstance(circuit, QuantumScript):
        return _tape_qasm3(
            circuit,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
            gateset=gateset,
        )

    @wraps(circuit)
    def wrapper(*args, **kwargs) -> str:
        tape = construct_tape(circuit)(*args, **kwargs)
        return _tape_qasm3(
            tape,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
            gateset=gateset,
        )

    return wrapper
