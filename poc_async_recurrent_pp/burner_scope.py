"""Detect only the user's explicitly scoped project burner processes."""
import os
from pathlib import Path


def pending_burner(gpus):
    requested=set(gpus)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid!=os.getuid():
                continue
            words=[part.decode() for part in (process/'cmdline').read_bytes().split(b'\0') if part]
            if '/home/esjung/vllm-ep/utilize.py' in words:
                if '--gpu' in words:
                    devices={int(words[words.index('--gpu')+1])}
                elif '--gpus' in words:
                    devices=set(map(int,words[words.index('--gpus')+1].split(',')))
                else:
                    continue
            elif '/home/esjung/vllm-ep/run_utilize.sh' in words:
                fields=dict(part.decode().split('=',1) for part in (process/'environ').read_bytes().split(b'\0') if b'=' in part)
                devices=set(map(int,fields.get('VLLM_UTILIZE_GPUS','').split(',')))
            else:
                continue
            # A multi-device supervisor does not respawn a stopped worker.
            # It must not stand in for a missing single-device worker.
            if requested==devices:
                return int(process.name)
        except (FileNotFoundError,PermissionError,ValueError,IndexError):
            continue
    return None
