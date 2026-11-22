"""Resume a frozen queue after an identified controller releases its GPU pair."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from .common import ROOT, REPO, PYTHON, dispatch_allowed, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('controller')
    parser.add_argument('manifest')
    args = parser.parse_args()
    receipt = json.loads((ROOT / 'state' / ('background_' + args.controller + '.json')).read_text())
    pid = receipt['pid']
    expected = receipt['command'][3].encode()
    while True:
        assert dispatch_allowed()
        proc = Path('/proc') / str(pid)
        if not proc.exists():
            break
        command = (proc / 'cmdline').read_bytes()
        if not command:
            break
        assert expected in command, 'Controller PID changed ownership'
        time.sleep(5)
    print('PREDECESSOR_EXITED', args.controller, 'RESUME', args.manifest, flush=True)
    code = subprocess.run([str(PYTHON), '-u', '-m',
        'poc_flashvep.dense_ar_longbench400_pd.queue', args.manifest], cwd=REPO).returncode
    write(ROOT / 'state' / ('after_' + Path(args.manifest).stem + '.json'),
          dict(time=time.time(), predecessor=args.controller, returncode=code,
               scope='A predecessor exit is not evidence that its research passed. Preserve its own gate and failures.'))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
