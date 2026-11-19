# CPU tooling recovery

The draft report builder encountered `KeyError: stage`.
The first two journal entries use `phase` and older field names.
The builder now accepts both schemas. The original journal stays unchanged.

The first final-check attempt could not serialize a `Path` in its CPU-test command receipt.
The command now stores a string. The repeated check passed all44 CPU tests and the measured gates.
This recovery did not rerun GPU experiments or change their raw results.

The inherited-source audit initially included the generated top-level `sitecustomize.py` bootstrap.
That bootstrap is not an inherited Git file.
The audit now checks inherited `poc_flashvep` Python sources only.
All742 inherited files equal the declared base and the current working copies.

These are CPU tooling failures. They are not additional failed Serving experiments.
The initial Native v1 configuration failure remains separate and preserved.
