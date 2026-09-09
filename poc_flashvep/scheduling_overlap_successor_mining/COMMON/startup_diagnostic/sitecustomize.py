"""Opt-in startup diagnostics only; exclude this path from measured runs."""
import faulthandler
import os

if os.environ.get("SUCCESSOR_STARTUP_DIAGNOSTIC") == "1":
    faulthandler.enable()
    faulthandler.dump_traceback_later(75, repeat=True)
