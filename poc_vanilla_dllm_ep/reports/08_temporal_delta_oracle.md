# Temporal delta-dispatch oracle

An exact dense delta has the same 2,048 FP16 elements as the original hidden
vector; subtraction alone saves no bytes. Approximate FP8/sparse deltas are
reported only as codec-free upper bounds and are killed if even that bound is
small or if temporal change is not compressible.


Lag-1 hidden relative L2 is 27.6%, and even stable routed branches change their
expert contribution by 32.4% relative L2. Sending an exact dense delta therefore
changes values but not the 2,048-element payload: the exact byte and E2E saving
is 0%. An impossible free FP8-delta bound—half the communication bytes, with no
codec or quality cost—reaches only 3.11% of clean request time after observer-tax
scaling.

**Decision: KILL (<5% even for the approximate codec-free cap).**
