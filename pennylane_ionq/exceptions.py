# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This file contains the definition of several custom exceptions raised in this module.
"""


class CircuitIndexNotSetException(Exception):
    """Raised after submitting multiple circuits if the circuit index is not set
    before the user want to access implementation methods of ``IonQDevice``
    like ``probability()``, ``estimate_probability()``, ``sample()`` or the
    ``prob`` property.
    """

    def __init__(self):
        self.message = (
            "Because multiple circuits have been submitted in this job, the index of the circuit "
            "you want to access must be first set via the set_current_circuit_index device method."
        )
        super().__init__(self.message)


class NotSupportedEvolutionInstance(Exception):
    """Raised when Evolution operation generator is not yet supported and is not converted to
    pauliexp IonQ gate.
    """

    def __init__(self):
        self.message = "The current instance of the Evolution gate is not supported."
        super().__init__(self.message)


class OperatorNotSupportedInEvolutionGateGenerator(Exception):
    """Raised when an Evolution gate is generated from a generator constructed with
    an operator that is not supported.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ComplexEvolutionCoefficientsNotSupported(Exception):
    """Raised when a coefficient in Evolution gate is complex."""

    def __init__(self):
        self.message = "Complex coefficients in Evolution gates are not supported."
        super().__init__(self.message)
