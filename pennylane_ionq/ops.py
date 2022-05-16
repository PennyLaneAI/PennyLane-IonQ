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
Custom operations
"""
from pennylane.operation import Operation

# Custom operations for the native gateset below.
class GPI(Operation):  # pylint: disable=too-few-public-methods
    r"""GPI(phi, wires)
    Single-qubit GPI gate.

    .. math::

       GPI(\phi) =
            \begin{pmatrix}
                0 & e^{-i 2 \pi \phi} \\
                e^{i 2 \pi \phi} & 0
            \end{pmatrix}
    Args:
        phi (float): phase :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 1
    num_wires = 1
    grad_method = None


class GPI2(Operation):  # pylint: disable=too-few-public-methods
    r"""GPI2(phi, wires)
    Single-qubit GPI2 gate.

    .. math::

       GPI2(\phi) =
            \begin{pmatrix}
                 1 & -i e^{-2 \pi i \phi} \\
                 -i e^{2 \pi i \phi} & 1
            \end{pmatrix}
    Args:
        phi (float): phase :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 1
    num_wires = 1
    grad_method = None


class MS(Operation):  # pylint: disable=too-few-public-methods
    r"""MS(phi0, phi1, wires)
    2-qubit entanlging MS gate.

    .. math::

       MS(\phi_{0}, \phi_{1}) =
            \frac{1}{\sqrt{2}}\begin{pmatrix}
                1 & 0 & 0 & -i e^{-2 \pi i(\phi_{0}+\phi_{1})} \\
                0 & 1 & -i e^{-2 \pi i (\phi_{0}-\phi_{1})} & 0 \\
                0 & -i e^{2 \pi i(\phi_{0}-\phi_{1})} & 1 & 0 \\
                -i e^{2 \pi i(\phi_{0}+\phi_{1})} & 0 & 0 & 1
            \end{pmatrix}
    Args:
        phi0 (float): phase of the first qubit :math:`\phi`
        phi1 (float): phase of the second qubit :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 2
    num_wires = 2
    grad_method = None


# Custom operations for the QIS Gateset below


class XX(Operation):
    r"""XX(phi, wires)
    The Ising XX gate.

    .. math:: XX(\phi) = e^{-\frac{\phi}{2}\hat{X}\otimes\hat{X}}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(XX(\phi)) = \frac{1}{2}\left[f(XX(\phi+\pi/2)) - f(XX(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`XX(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"


class YY(Operation):
    r"""YY(phi, wires)
    The Ising YY gate.

    .. math:: YY(\phi) = e^{-\frac{\phi}{2}\hat{Y}\otimes\hat{Y}}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(YY(\phi)) = \frac{1}{2}\left[f(YY(\phi+\pi/2)) - f(YY(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`YY(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"


class ZZ(Operation):
    r"""ZZ(phi, wires)
    The Ising ZZ gate.

    .. math:: ZZ(\phi) = e^{-\frac{\phi}{2}\hat{Z}\otimes\hat{Z}}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(ZZ(\phi)) = \frac{1}{2}\left[f(ZZ(\phi+\pi/2)) - f(ZZ(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`ZZ(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the subsystems the operation acts on
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"
