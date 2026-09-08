"""Opt-in worker hook for paper-baseline transfer, never installed globally."""
import os

if os.environ.get("SUCCESSOR_EP_ENABLE")=="1":
    from successor_hook import install
    install()
