"""Give long controllers a durable log that survives an API-server restart."""
import argparse
import subprocess
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--after', type=int,
        help='Wait for this sequence owner to close before starting the block')
    args = parser.parse_args()
    initialize()
    stamp = time.time_ns()
    log_path = ROOT/'logs'/f'controller_{stamp}.log'
    command = (['poc_flashvep.huginn_net_benefit.after_sequence', str(args.after), *args.names]
        if args.after else ['poc_flashvep.huginn_net_benefit.sequence', '--independent', *args.names])
    with log_path.open('ab', buffering=0) as output:
        process = subprocess.Popen(['/home/esjung/.venvs/looped-pd-abc/bin/python', '-u', '-m', *command],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT/'state'/f'detached_controller_{stamp}.json', dict(pid=process.pid,
        time=time.time(), names=args.names, durable_log=str(log_path), previous_owner_pid=args.after,
        scope='The sequential owner writes to a file rather than the API connection. The normal GPU guard, timeouts, qualification gates, and final restoration remain enabled.'))
    print(json.dumps(dict(pid=process.pid, log=str(log_path), jobs=len(args.names), after=args.after)))


if __name__ == '__main__':
    main()
