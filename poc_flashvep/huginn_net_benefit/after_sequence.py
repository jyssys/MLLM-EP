"""Start the next serial block only after the previous owner restores resources."""
import argparse
import subprocess
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('previous_pid', type=int)
    parser.add_argument('names', nargs='+')
    parser.add_argument('--maximum-wait-seconds', type=int, default=7200)
    args = parser.parse_args()
    initialize()
    receipt = ROOT/'state'/f'pending_sequence_{os.getpid()}.json'
    start = time.time()
    write(receipt, dict(status='WAITING_PREVIOUS_RESOURCE_OWNER',
        previous_pid=args.previous_pid, names=args.names, time=start))
    while True:
        matches = list((ROOT/'state').glob(f'sequence_{args.previous_pid}_*.json'))
        assert len(matches) == 1, 'The previous sequence history must be unique'
        previous = json.loads(matches[0].read_text())
        if previous['status'] == 'CLOSED':
            break
        os.kill(args.previous_pid, 0)
        if time.time()-start > args.maximum_wait_seconds:
            raise TimeoutError('The previous resource owner did not close within the explicit waiting budget')
        time.sleep(2)
    write(receipt, dict(status='STARTING_SERIAL_BLOCK', previous_pid=args.previous_pid,
        names=args.names, time=time.time()))
    value = subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python', '-m',
        'poc_flashvep.huginn_net_benefit.sequence', '--independent', *args.names], cwd=REPO)
    write(receipt, dict(status='CLOSED', previous_pid=args.previous_pid,
        names=args.names, returncode=value.returncode, time=time.time()))
    raise SystemExit(value.returncode)


if __name__ == '__main__':
    main()
