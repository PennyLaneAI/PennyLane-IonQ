# Release 0.20.0-dev

### New features since last release

### Breaking changes

### Improvements

### Documentation

### Bug fixes

* If response returns as failure, an error is raised. If the user
  submits an empty circuit, a warning is raised.
  [(#43)](https://github.com/PennyLaneAI/PennyLane-IonQ/pull/43)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee

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
