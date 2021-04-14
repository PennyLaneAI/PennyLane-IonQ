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


class XX(Operation):
    r"""XX(phi, wires)
    The Ising XX gate.

    .. math:: XX(\phi) = \frac{1}{\sqrt{2}}\begin{bmatrix}
            1 & 0 & 0 & -ie^{i\phi} \\
            0 & 1 & -i & 0\\
            0 & -i & 1 & 0\\
            -ie^{-i\phi} & 0 & 0 & 1
        \end{bmatrix}.

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
    par_domain = "R"
    grad_method = "A"


class YY(Operation):
    r"""YY(phi, wires)
    The Ising YY gate.

    .. math:: YY(\phi) = \begin{bmatrix}
            \cos(\phi) & 0 & 0 & i\sin(\phi) \\
            0 & \cos(\phi) & -i\sin(\phi) & 0\\
            0 & -i\sin(\phi) & \cos(\phi) & 0\\
            i\sin(\phi) & 0 & 0 & \cos(\phi)
        \end{bmatrix}.

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
    par_domain = "R"
    grad_method = "A"


class ZZ(Operation):
    r"""ZZ(phi, wires)
    The Ising ZZ gate.

    .. math:: ZZ(\phi) = \begin{bmatrix}
            e^{i \phi/2} & 0 & 0 & 0 \\
            0 & e^{-i \phi/2} & 0 & 0\\
            0 & 0 & e^{-i \phi/2} & 0\\
            0 & 0 & 0 & e^{i \phi/2}
        \end{bmatrix}.

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
    par_domain = "R"
    grad_method = "A"
