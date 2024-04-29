# Release 0.36.0

### Contributors ✍️

This release contains contributions from (in alphabetical order):

---
# Release 0.34.0

### New features since last release

* Application of debiasing and sharpening for error mitigation is made available, with parameters set on device initialization. Error mitigation strategies that 
  need to be set at runtime are defined in the `error_mitigation` dictionary (currently a single strategy, `debias`, is available). Whether or not to
  apply sharpening to the returned results is set via the parameter `sharpen`. A device using debiasing and sharpening to mitigate errors can be initialized as:
  
  ```python
  import pennylane as qml

  dev = qml.device("ionq.qpu", wires=2, error_mitigation={"debias": True}, sharpen=True)
  ```

  For more details, see the [IonQ Guide on sharpening and debiasing](https://ionq.com/resources/debiasing-and-sharpening), or refer to the publication <https://arxiv.org/pdf/2301.07233.pdf>
  [(#75)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/75)
  [(#96)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/96)

### Improvements 🛠

* The IonQ API version accessed via the plugin is updated from 0.1 to 0.3
  [(#75)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/75)
  [(#96)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/96)
  
* Use new `backend` field to specify `qpu`.
  [(#81)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/81)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Spencer Churchill
Lillian Frederiksen

---
# Release 0.32.0

### Breaking changes 💔

* Support for Python 3.8 has been removed, and support for 3.11 has been added.
  [(#78)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/78)

### Improvements 🛠

* Added support for `qml.StatePrep` as a state preparation operation.
  [(#77)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/77)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Mudit Pandey,
Jay Soni

---
# Release 0.28.0

### New features since last release

* Add support for various IonQ native gates.
  [(#55)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/55)

### Contributors

This release contains contributions from (in alphabetical order):

Jon Donovan

---
# Release 0.23.0

### Improvements

* Added high level access to the `target` kwarg in the 
  `SimulatorDevice` class for general IonQ devices.
  [(#50)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/50)

### Bug Fixes

* Since the histogram of probabilities returned from the remote simulator does not always sum exactly to one,
  the PennyLane device normalizes them to higher precision.
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
