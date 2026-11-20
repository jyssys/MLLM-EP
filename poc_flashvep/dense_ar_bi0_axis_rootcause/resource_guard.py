"""Stop only approved burner sessions. Ignore closeout metadata files."""
import json
import os
from pathlib import Path
import signal
import time


def supervisor_record(value):
    return isinstance(value,dict) and isinstance(value.get('GPUs'),list) and isinstance(value.get('supervisor'),int)


def stop(owner,gpus):
    owned=owner.burners()
    before=[r for r in owner.processes() if r['gpu'] in gpus]
    for row in before:
        assert Path('/proc',str(row['pid'])).stat().st_uid==os.getuid()
        assert os.getsid(row['pid'])==owned.get(row['gpu']),('UNRELATED_JOB_LEFT_UNTOUCHED',row)
    owner.append(owner.ROOT/'resource_ledger.jsonl',dict(action='STOP_OWNED_BURNERS',time=time.time(),GPUs=gpus,owned=owned,active=before))
    files=list((owner.ROOT/'state').glob('burner_*.json'))+list((owner.PRIOR/'state').glob('burner_*.json'))
    for path in files:
        record=json.loads(path.read_text())
        if not supervisor_record(record) or not set(record['GPUs'])<=set(gpus):
            continue
        pid=record['supervisor'];proc=Path('/proc',str(pid))
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and owner.SCRIPT in words and '--worker' not in words:
                os.kill(pid,signal.SIGTERM);time.sleep(.2)
                if proc.exists() and (proc/'cmdline').read_bytes():
                    os.kill(pid,signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):
            pass
    for gpu,pid in owner.burners().items():
        if gpu in gpus:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    until=time.monotonic()+45
    while (any(r['gpu'] in gpus for r in owner.processes()) or set(owner.burners())&set(gpus)) and time.monotonic()<until:
        time.sleep(.5)
    assert not [r for r in owner.processes() if r['gpu'] in gpus]
    assert not set(owner.burners())&set(gpus)
