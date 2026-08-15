# model/

No trained model weights are included, because none are used.

Drift-Sense is a fully classical computer-vision pipeline — coarse-to-fine
normalized cross-correlation over scale and rotation, periodicity-suppressed
fingerprint verification, and parabolic sub-pixel refinement (see
**"How the localization algorithm works"** in the top-level `README.md`).
There is no training step, no learned parameters, and no pretrained model
dependency of any kind.

This folder is kept empty (rather than omitted) only so the submission's
directory listing matches the challenge's recommended layout one-to-one,
and so the absence of weights is stated explicitly rather than left
ambiguous.
