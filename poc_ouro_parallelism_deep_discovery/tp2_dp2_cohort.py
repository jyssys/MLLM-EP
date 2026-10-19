"""CPU protocol for two independent native TP2 groups. No model code runs here."""
import os
from pathlib import Path
import time

from poc_ouro_parallelism_deep_discovery.contract import GPUS, UUIDS
from poc_ouro_parallelism_deep_discovery.dp4_cohort import collect_messages, serve_worker


GROUPS = ((4, 5), (6, 7))
REPLICAS = 2


def validate_groups(groups):
    if groups != [list(group) for group in GROUPS]:
        raise ValueError('Require exactly physical groups [4,5] and [6,7].')
    return True


def assignments(case):
    requests = case['requests']
    if len(requests) != case['C'] or [r['cohort_slot'] for r in requests] != list(range(case['C'])):
        raise ValueError('The cohort slots must be unique and contiguous.')
    groups = [[] for _ in range(REPLICAS)]
    for request in requests:
        if len(request['prompt_token_ids']) != case['context']:
            raise ValueError('A frozen prompt has the wrong context length.')
        groups[request['cohort_slot'] % REPLICAS].append(request)
    return groups


def validate_ownership(group, layers, ownership):
    if tuple(group) not in GROUPS or len(ownership) != 2:
        raise ValueError('A TP2 group requires its two declared rank records.')
    by_rank = {row['rank']: row for row in ownership}
    if set(by_rank) != {0, 1} or len(by_rank) != len(ownership):
        raise ValueError('TP2 ownership needs distinct ranks 0 and 1.')
    pids = []
    for rank, physical_gpu in enumerate(group):
        row = by_rank[rank]
        expected_uuid = UUIDS[GPUS.index(physical_gpu)]
        if (row['world'] != 2 or row['GPU_uuid'].removeprefix('GPU-') != expected_uuid
                or row['CUDA_VISIBLE_DEVICES'] != ','.join(map(str, group))
                or row['device'] != f'cuda:{rank}' or row['cache_slots'] != 4 * layers
                or row['physical_layers'] != layers):
            raise ValueError('A TP2 rank has the wrong device, world, or cache identity.')
        if len(row['geometry']) != layers or any(
                item['Q_heads'] != 8 or item['KV_heads'] != 8 or item['head_dim'] != 128
                or item['loop_cache_modules'] != 4 for item in row['geometry']):
            raise ValueError('TP2 requires eight local Q/KV heads and four depth banks.')
        if type(row['pid']) is not int or row['pid'] <= 0:
            raise ValueError('A TP2 rank needs its actual positive PID.')
        pids.append(row['pid'])
    if len(set(pids)) != 2:
        raise ValueError('TP2 rank processes must have distinct PIDs.')
    return True


def inspect_process_leases(pids, pgid, sid, guard_output, ancestor_pid=None):
    """Inspect only declared child PIDs. Do not signal any process."""
    records = []
    for pid in pids:
        process = Path('/proc') / str(pid)
        if (process.stat().st_uid != os.getuid() or os.getpgid(pid) != pgid
                or os.getsid(pid) != sid):
            raise RuntimeError('A declared process escaped the owned session.')
        environment = process.joinpath('environ').read_bytes().split(b'\0')
        if ('LOOPED_KV_OUTPUT=' + guard_output).encode() not in environment:
            raise RuntimeError(f'A declared process {pid} lost its guard output identity.')
        ancestry = []
        if ancestor_pid is not None:
            current = pid
            for _ in range(64):
                if current in ancestry:
                    raise RuntimeError('A process ancestry cycle prevents lease validation.')
                ancestry.append(current)
                current_process = Path('/proc') / str(current)
                if (current_process.stat().st_uid != os.getuid() or os.getpgid(current) != pgid
                        or os.getsid(current) != sid):
                    raise RuntimeError('A process ancestor escaped the owned session.')
                if current == ancestor_pid:
                    break
                fields = [line.split()[1] for line in current_process.joinpath('status').read_text().splitlines()
                          if line.startswith('PPid:')]
                if len(fields) != 1 or int(fields[0]) <= 0:
                    raise RuntimeError('A declared rank lacks its frontend ancestor.')
                current = int(fields[0])
            else:
                raise RuntimeError('The process ancestry exceeds the bounded lease check.')
            if ancestry[-1] != ancestor_pid:
                raise RuntimeError('A declared rank lacks its frontend ancestor.')
        records.append(dict(pid=pid, pgid=pgid, sid=sid, guard_output_matches=True,
                            ancestor_pid=ancestor_pid, ancestry_to_frontend=ancestry))
    return records


def aggregate(case, results, length, released_ns, completed_ns):
    expected = assignments(case)
    if sorted(results) != list(range(REPLICAS)) or completed_ns <= released_ns:
        raise ValueError('Both completions and a positive parent interval are required.')
    duration = completed_ns - released_ns
    workers = []
    for worker in range(REPLICAS):
        value = results[worker]
        slots = [r['cohort_slot'] for r in expected[worker]]
        records = value['request_records']
        expected_sources = [request['source_request_id'] for request in expected[worker]]
        if (value['worker'] != worker or value['physical_gpus'] != list(GROUPS[worker])
                or value['assigned_slots'] != slots
                or value['generated_tokens'] != len(slots) * length or len(records) != len(slots)
                or [r['cohort_slot'] for r in records] != slots
                or [r['source_request_id'] for r in records] != expected_sources
                or any(r['output_tokens'] != length for r in records)
                or any(r['finish_reason'] != 'length' for r in records)
                or value['output_identity'] != 'ALL_FROZEN_CONSTANT_TOKEN_IDS'):
            raise ValueError('A completion has the wrong group, slots, or output denominator.')
        wall = value['local_generate_wall_ns']
        if not 0 <= wall <= duration or (bool(slots) != (wall > 0)):
            raise ValueError('The local generate interval is invalid for this assignment.')
        received = value['release_received_ns']
        start, end = value['local_generate_start_ns'], value['local_generate_end_ns']
        if type(received) is not int or not released_ns <= received <= completed_ns:
            raise ValueError('A release receipt lies outside the parent interval.')
        if slots:
            if (type(start) is not int or type(end) is not int
                    or not received <= start <= end <= completed_ns or end - start != wall):
                raise ValueError('An active local interval has inconsistent endpoints.')
        elif start is not None or end is not None:
            raise ValueError('An idle group must have no local generate endpoints.')
        workers.append(value)
    total = case['C'] * length
    return dict(global_makespan_ns=duration, global_makespan_ms=duration / 1e6,
                generated_tokens=total, cohort_output_TPS=total * 1e9 / duration,
                release_start_ns=released_ns, all_completed_ns=completed_ns,
                active_replicas=sum(bool(group) for group in expected), reserved_replicas=2,
                reserved_gpus=4, tensor_parallel_size_per_replica=2, workers=workers,
                metric='Parent release-to-both-completed wall. Includes IPC and controller work.')


def run_cohort(connections, case, case_index, length, lane, trial_id, timeout, transcript):
    if sorted(connections) != list(range(REPLICAS)):
        raise ValueError('Require both TP2 frontend connections.')
    groups = assignments(case)
    command = dict(kind='PREPARE', trial_id=trial_id, case_index=case_index, length=length, lane=lane)
    for connection in connections.values():
        connection.send(command)
    ready = collect_messages(connections, 'READY', timeout, trial_id, transcript)
    for worker, message in ready.items():
        if message.get('assigned_slots') != [r['cohort_slot'] for r in groups[worker]]:
            raise RuntimeError('A TP2 frontend prepared the wrong cohort slots.')
    released_ns = time.perf_counter_ns()
    for connection in connections.values():
        connection.send(dict(kind='RELEASE', trial_id=trial_id))
    done = collect_messages(connections, 'DONE', timeout, trial_id, transcript)
    completed_ns = time.perf_counter_ns()
    result = aggregate(case, {worker: message['result'] for worker, message in done.items()},
                       length, released_ns, completed_ns)
    return dict(result, trial_id=trial_id, lane=lane)
