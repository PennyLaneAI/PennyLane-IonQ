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
=================

**Module name:** :mod:`pennylane_ionq.ops`

.. currentmodule:: pennylane_ionq.ops

This module contains some additional IonQ qubit operations.

They can be imported via

.. code-block:: python

    from pennylane_ionq.ops import S, T, CCNOT

Operations
----------

.. autosummary::
    S
    Sdg
    T
    Tdg
    CCNOT
    V
    Vdg
    XX
    YY
    ZZ


Code details
~~~~~~~~~~~~
"""
from pennylane.operation import Operation


class S(Operation):
    r"""S(wires)
    S gate.

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class Sdg(Operation):
    r"""Sdg(wires)
    :math:`S^\dagger` gate.

    .. math:: S^\dagger = \begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class T(Operation):
    r"""T(wires)
    T gate.

    .. math:: T = \begin{bmatrix}1&0\\0&e^{i \pi / 4}\end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class Tdg(Operation):
    r"""Tdg(wires)
    :math:`T^\dagger` gate.

    .. math:: T^\dagger = \begin{bmatrix}1&0\\0&e^{-i \pi / 4}\end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class CCNOT(Operation):
    r"""CCNOT(wires)
    Controlled-controlled-not gate.

    .. math::

        CCNOT = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{bmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None


class V(Operation):
    r"""V(wires)
    V gate.

    .. math::

        V = \sqrt{X} = \frac{1}{2}\begin{bmatrix}
            1+i & 1-i \\
            1-i & 1+i \\
        \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class Vdg(Operation):
    r"""Vdg(wires)
    :math:`V^\dagger` gate.

    .. math::

        V^\dagger = \sqrt{X}^\dagger = \frac{1}{2}\begin{bmatrix}
            1-i & 1+i \\
            1+i & 1-i \\
        \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


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
        wires (Sequence[int] or int): the wire the operation acts on
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
        wires (Sequence[int] or int): the wire the operation acts on
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
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
