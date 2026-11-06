"""Record budget and dispatch transitions without modifying GPU ownership."""
import os
import time
from pathlib import Path
from .common import *


def jobs():
    selected=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            words=(path/'cmdline').read_bytes().decode().split('\0')
            if '-m' not in words:continue
            module=words[words.index('-m')+1]
            if module.startswith('poc_flashvep.d_local_cross_depth_batching.') and module.rsplit('.',1)[1] not in ('deadline_guard','watch_campaign'):
                selected.append(dict(pid=int(path.name),module=module))
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,UnicodeDecodeError):pass
    return selected


def main():
    previous=None;empty=0
    path=ROOT/'state/time_budget.jsonl'
    while time.time()<DEADLINE:
        current=jobs();active=ROOT/'state/active.json'
        value=json.loads(active.read_text()) if active.exists() else {}
        row=dict(time=time.time(),elapsed_hours=(time.time()-START)/3600,
            GPU_budget_remaining_hours=(GPU_DEADLINE-time.time())/3600,active=value,controllers=current)
        with path.open('a') as out:out.write(json.dumps(row)+'\n')
        key=value.get('name',value.get('status'))
        if key!=previous:
            print(json.dumps(dict(time=time.time(),active=key,controllers=len(current))),flush=True);previous=key
        empty=empty+1 if not current else 0
        if empty>=2:
            print('ALL_REGISTERED_CONTROLLERS_FINISHED',flush=True);return
        time.sleep(15)


if __name__=='__main__':main()
