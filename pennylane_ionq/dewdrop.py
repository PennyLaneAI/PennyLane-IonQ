# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Base Framework device class
===========================

**Module name:** :mod:`plugin_name.device`

.. currentmodule:: plugin_name.device

An abstract base class for constructing Target Framework devices for PennyLane.

This should contain all the boilerplate for supporting PennyLane
from the Target Framework, making it easier to create new devices.
The abstract base class below should contain all common code required
by the Target Framework.

This abstract base class will not be used by the user. Add/delete
methods and attributes below where needed.

See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of how the Device class works.

Classes
-------

.. autosummary::
   FrameworkDevice

Code details
~~~~~~~~~~~~
"""
import itertools
import functools
from time import sleep

# we always import NumPy directly
import numpy as np

from pennylane import Device

from .api_client import Job
from ._version import __version__


class DewdropDevice(Device):
    r"""Abstract Framework device for IonQ Dewdrop.

    Args:
        wires (int): the number of modes to initialize the device in
        target (str): the target device, either ``"simulator"`` or ``'qpu'``
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
    """
    name = 'IonQ Dewdrop PennyLane plugin'
    pennylane_requires = '>=0.5.0'
    version = __version__
    author = 'XanaduAI'

    short_name = 'ionq.dewdrop'
    _operation_map = {
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
        # operations not natively implemented in IonQ but provided in gates.py
        # "Rot": Rot,
        # "BasisState": BasisState,
        # additional operations not native to PennyLane but present in IonQ
        "S": "s",
        "Sdg": "si",
        "T": "t",
        "Tdg": "ti",
        "V": "v",
        "Vdg": "vi",
        "XX": "xx",
        "YY": "YY",
        "ZZ": "zz",
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}

    def __init__(self, wires, *, target="simulator", shots=1024):
        super().__init__(wires, shots)
        self.target = target
        self.reset()

    def reset(self):
        """Reset the device"""
        self.prob = {}
        self.circuit = {"qubits": self.num_wires, "circuit": []}
        self.job = {"lang": "json", "body": self.circuit, "target": self.target}

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operation, wires, par):
        op_name = self._operation_map[operation]
        gate = {"gate": op_name, "target": wires[0]}

        if len(wires) == 2:
            gate["control"] = wires[1]

        if par:
            gate["rotation"] = par[0]

        self.circuit["circuit"].append(gate)

    def post_apply(self):
        for e in self.obs_queue:
            # Add unitaries prior to measurement if the expectation
            # depending on the observable to be measured
            self.rotate_basis(e.name, e.wires, e.parameters)

    def pre_measure(self):
        job = Job()

        # send job for exection
        job.manager.create(**self.job)

        # retrieve results
        while not job.is_complete:
            sleep(0.01)
            job.reload()

        job.manager.get(job.id.value)

        histogram = job.data.value["histogram"]
        self.prob = np.array([histogram.get(str(k), 0) for k in range(2**self.num_wires)])

    def expval(self, observable, wires, par):
        eigvals = self.eigvals(observable, wires, par)
        return (eigvals @ self.prob).real

    def var(self, observable, wires, par):
        eigvals = self.eigvals(observable, wires, par)
        return (eigvals ** 2) @ self.prob - (eigvals @ self.prob).real ** 2

    def rotate_basis(self, observable, wires, par):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (str): the name of an observable
            wires (List[int]): wires the observable is measured on
            par (List[Any]): parameters of the observable
        """
        if observable == "PauliX":
            # X = H.Z.H
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "PauliY":
            # Y = (HS^)^.Z.(HS^) and S^=SZ
            self.apply("PauliZ", wires=wires, par=[])
            self.apply("S", wires=wires, par=[])
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "Hadamard":
            # H = Ry(-pi/4)^.Z.Ry(-pi/4)
            self.apply("RY", wires, [-np.pi / 4])

    def eigvals(self, observable, wires, par):
        """Determine the eigenvalues of observable(s).

        Args:
            observable (str, List[str]): the name of an observable,
                or a list of observables representing a tensor product.
            wires (List[int]): wires the observable(s) is measured on
            par (List[Any]): parameters of the observable(s)

        Returns:
            array[float]: an array of size ``(len(wires),)`` containing the
            eigenvalues of the observable
        """
        # observable should be Z^{\otimes n}
        eigvals = z_eigs(len(wires))

        if isinstance(observable, list):
            # determine the eigenvalues
            if "Hermitian" in observable:
                # observable is of the form Z^{\otimes a}\otimes H \otimes Z^{\otimes b}
                eigvals = np.array([1])

                for k, g in itertools.groupby(zip(observable, wires, par), lambda x: x[0] == "Hermitian"):
                    if k:
                        p = list(g)[0][2]
                        Hkey = tuple(p[0].flatten().tolist())
                        eigvals = np.kron(eigvals, self._eigs[Hkey]["eigval"])
                    else:
                        n = len([w for sublist in list(zip(*g))[1] for w in sublist])
                        eigvals = np.kron(eigvals, z_eigs(n))

        elif observable == "Hermitian":
            # single wire Hermitian observable
            Hkey = tuple(par[0].flatten().tolist())
            eigvals = self._eigs[Hkey]["eigval"]

        elif observable == "Identity":
            eigvals = np.ones(2 ** len(wires))

        return eigvals


@functools.lru_cache()
def z_eigs(n):
    r"""Returns the eigenvalues for :math:`Z^{\otimes n}`.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}`
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([z_eigs(n - 1), -z_eigs(n - 1)])
