# Nsight Findings

Nsight Systems was not escalated for this PoC. Clean same-device CUDA-event measurements already gave a decisive result with no unexplained tail or ambiguous stage attribution:

- ideal block overlap oracle was large;
- real block streaming never beat its same-chunk/no-overlap control by more than 1.04%;
- clean timing without per-block stage events produced −0.00% to −0.78% streaming benefit;
- both chunked variants were substantially slower than one whole EP invocation.

An Nsight trace could illustrate the already-established launch/communication fragmentation, but it could not change the kill gate and would add observer perturbation to micro-invocations. This file records an intentional non-escalation, not missing evidence presented as a profile.
