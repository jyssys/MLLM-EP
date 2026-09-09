"""Opt-in, process-local observation only; never installed in a baseline env."""
import os

if os.environ.get("SCHEDULING_EP_OBSERVE") == "1":
    if os.environ.get("SCHEDULING_EP_OBSERVER_MODE") == "layer_sampled":
        from observer_lite import install
    else:
        from observer import install
    install()
