# Release 0.23.0

### Improvements

* Added high level access to the `target` kwarg in the 
  `SimulatorDevice` class for general IonQ devices.
  [(#50)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/50)

### Bug Fixes

* Since the probabilities returned don't always perfectly sum to one, they are now normalized.
  [(#53)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/53)

### Contributors

This release contains contributions from (in alphabetical order):

Jon Donovan, Christina Lee, Antal Száva

---

# Release 0.20.0

### Improvements

* Added support for Python 3.10.
  [(#46)](https://github.com/PennyLaneAI/pennylane-forest/pull/46)

### Bug fixes

* Parameters are converted to floats, unwrapping interface data types.
  [(#41)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/41)

* If response returns as failure, an error is raised. If the user
  submits an empty circuit, a warning is raised.
  [(#43)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/43)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Jay Soni, Antal Száva

---

# Release 0.16.0

### Improvements

* Return samples from the `QPUDevice` directly instead of resample from the returned results.
  [(#32)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/32)

### Contributors

This release contains contributions from (in alphabetical order):

Dave Bacon, Nathan Killoran

---

# Release 0.15.3

### Bug fixes

* Fixes a bug where the shot number was not correctly being submitted
  to the API.
  [(#29)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/27)

### Contributors

This release contains contributions from (in alphabetical order):

Dave Bacon, Josh Izaac.

---

# Release 0.15.2

### Improvements

* Adds a default timeout to requests of ten minutes. This is a timeout both on connect and request.
  [(#27)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/27)

### Contributors

This release contains contributions from (in alphabetical order):

Dave Bacon

---

# Release 0.15.1

### Improvements

* The IonQ plugin now uses the v0.1 endpoint for the IonQ API.
  [(#24)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/24)

### Contributors

This release contains contributions from (in alphabetical order):

Nathan Killoran

---

# Release 0.15.0

Initial release.

This release contains contributions from (in alphabetical order):

Josh Izaac, Nathan Killoran.
