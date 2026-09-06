# Nsight validation

The preceding fixed-shape root-cause artifact on the same runtime captured
child workers and found `deep_ep::intranode::notify_dispatch` up to 98.97 ms
and `deep_ep::intranode::barrier` up to 495.56 ms.  This follow-up uses the
validated Nsight evidence as corroboration and does not rerun the expensive
capture.  New measurements use same-device CUDA events and do not subtract
clocks across GPUs.
