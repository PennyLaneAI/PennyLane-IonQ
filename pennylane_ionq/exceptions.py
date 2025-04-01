class CircuitIndexNotSetException(Exception):
    """Raised when after submitting multiple circuits circuit index is not set
    before the user want to access implementation methods of IonQDevice
    like probability(), estimate_probability(), sample() or the prob property.
    """

    def __init__(self):
        self.message = (
            "Because multiple circuits have been submitted in this job, the index of the circuit "
            "you want to access must be first set via the set_current_circuit_index device method."
        )
        super().__init__(self.message)


class NotSupportedEvolutionOperationGenerator(Exception):
    """Raised when Evolution operation generator is not yet supported and is not converted to
    pauliexp IonQ gate.
    """

    def __init__(self):
        super().__init__(self.message)


class ComplexEvolutionCoefficientsNotSupported(Exception):
    """Raised when a coeffcient in Evolution gate is complex."""

    def __init__(self):
        super().__init__(self.message)
