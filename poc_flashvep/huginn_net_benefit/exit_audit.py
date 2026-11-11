"""Audit process exit separately from delivered tokens and exact state."""
from .common import *


def main():
    initialize()
    rows = []
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():
            continue
        log = ROOT/'logs'/(folder.name+'.log')
        text = log.read_text(errors='replace') if log.exists() else ''
        faults = [token for token in ('Segfault encountered', 'Fatal Python error:',
            'CUDA error: an illegal memory access') if token in text]
        receipts = [json.loads(path.read_text()) for path in folder.glob('native_worker_exit_*.json')]
        native = [row for value in receipts for row in value['records']]
        roles = {row['role'] for row in native if row['exitcode'] == 0 and not row['forced_cleanup']}
        failed = bool(faults or any(row['exitcode'] != 0 for row in native))
        state = 'FAIL' if failed else 'VERIFIED_ZERO_EXIT' if roles == {'prefill', 'decode'} else 'NOT_VERIFIED_CHILD_EXIT'
        item = dict(name=folder.name, status=state, fault_signatures=faults,
            worker_exits=native,
            scope='A completed cohort and state equality do not prove a clean worker exit. Old no-fault logs alone do not establish exit code zero.')
        write(folder/'process_exit_audit.json', item)
        rows.append(item)
    write(ROOT/'audit/process_exit_inventory.json', rows)
    print(json.dumps(dict(cells=len(rows), failed=sum(row['status'] == 'FAIL' for row in rows),
        verified=sum(row['status'] == 'VERIFIED_ZERO_EXIT' for row in rows))))


if __name__ == '__main__':
    main()
