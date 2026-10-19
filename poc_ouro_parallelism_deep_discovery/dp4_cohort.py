"""CPU protocol and metrics for four independent native TP1 replicas."""
from multiprocessing.connection import wait
import time


REPLICAS = 4


def assignments(case):
    """Assign each request once. Preserve its cohort slot and frozen prompt."""
    requests = case['requests']
    if len(requests) != case['C'] or [r['cohort_slot'] for r in requests] != list(range(case['C'])):
        raise ValueError('The cohort slots must be unique and contiguous.')
    groups = [[] for _ in range(REPLICAS)]
    for request in requests:
        if len(request['prompt_token_ids']) != case['context']:
            raise ValueError('A frozen prompt has the wrong context length.')
        groups[request['cohort_slot'] % REPLICAS].append(request)
    return groups


def collect_messages(connections, kind, timeout, trial_id=None, transcript=None):
    """Receive one matching response from each worker under one shared deadline."""
    pending = dict(connections)
    results = {}
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'{kind} deadline expired. Pending workers: {sorted(pending)}')
        ready = wait(list(pending.values()), timeout=remaining)
        if not ready:
            raise TimeoutError(f'{kind} deadline expired. Pending workers: {sorted(pending)}')
        for connection in ready:
            worker = next(index for index, value in pending.items() if value is connection)
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f'Worker {worker} closed its control channel during {kind}.') from error
            if transcript is not None:
                transcript.append(dict(received_ns=time.perf_counter_ns(), worker=worker, message=message))
            if not isinstance(message, dict) or message.get('worker') != worker:
                raise RuntimeError(f'Worker {worker} sent an invalid identity: {message!r}')
            if message.get('kind') != kind or (trial_id is not None and message.get('trial_id') != trial_id):
                raise RuntimeError(f'Worker {worker} sent an unexpected {kind} response: {message!r}')
            results[worker] = message
            del pending[worker]
    return results


def aggregate(case, results, length, released_ns, completed_ns):
    """Use the observed parent makespan. Never infer it from worker timings."""
    expected = assignments(case)
    if sorted(results) != list(range(REPLICAS)) or completed_ns <= released_ns:
        raise ValueError('All four completions and a positive parent interval are required.')
    duration = completed_ns - released_ns
    workers = []
    for worker in range(REPLICAS):
        result = results[worker]
        slots = [r['cohort_slot'] for r in expected[worker]]
        if result['worker'] != worker or result['assigned_slots'] != slots:
            raise ValueError('A worker completion has the wrong assignment.')
        if result['generated_tokens'] != len(slots) * length or len(result['request_records']) != len(slots):
            raise ValueError('A worker completion has the wrong output denominator.')
        if [record['cohort_slot'] for record in result['request_records']] != slots:
            raise ValueError('Native request records have the wrong cohort slots.')
        if any(record['output_tokens'] != length for record in result['request_records']):
            raise ValueError('A native request record has the wrong output length.')
        if result['output_identity'] != 'ALL_FROZEN_CONSTANT_TOKEN_IDS':
            raise ValueError('A worker did not retain the constant-token assertion.')
        if not 0 <= result['local_generate_wall_ns'] <= duration:
            raise ValueError('A worker interval is outside the parent makespan.')
        if not slots and result['local_generate_wall_ns'] != 0:
            raise ValueError('An idle worker must not execute a generate call.')
        if slots and result['local_generate_wall_ns'] == 0:
            raise ValueError('An active worker requires an observed generate interval.')
        workers.append(result)
    total = case['C'] * length
    return dict(global_makespan_ns=duration, global_makespan_ms=duration / 1e6,
                generated_tokens=total, cohort_output_TPS=total * 1e9 / duration,
                release_start_ns=released_ns, all_completed_ns=completed_ns,
                active_replicas=sum(bool(group) for group in expected),
                reserved_replicas=REPLICAS, workers=workers,
                metric='Parent release-to-all-completed wall. Includes IPC and control overhead.')


def run_cohort(connections, case, case_index, length, lane, trial_id, timeout, transcript):
    """Prepare every worker before the timed release, including idle workers."""
    if sorted(connections) != list(range(REPLICAS)):
        raise ValueError('The parent requires all four worker connections.')
    groups = assignments(case)
    command = dict(kind='PREPARE', trial_id=trial_id, case_index=case_index, length=length, lane=lane)
    for connection in connections.values():
        connection.send(command)
    ready = collect_messages(connections, 'READY', timeout, trial_id, transcript)
    for worker, message in ready.items():
        if message.get('assigned_slots') != [r['cohort_slot'] for r in groups[worker]]:
            raise RuntimeError('A worker reported readiness for the wrong cohort slots.')
    released_ns = time.perf_counter_ns()
    for connection in connections.values():
        connection.send(dict(kind='RELEASE', trial_id=trial_id))
    done = collect_messages(connections, 'DONE', timeout, trial_id, transcript)
    completed_ns = time.perf_counter_ns()
    value = aggregate(case, {worker: message['result'] for worker, message in done.items()},
                      length, released_ns, completed_ns)
    return dict(value, trial_id=trial_id, lane=lane)


def serve_worker(connection, worker, prepare, execute):
    """Reject stale releases and repeated trials. Do not execute before release."""
    seen = set()
    while True:
        command = connection.recv()
        if command.get('kind') == 'STOP':
            return
        if command.get('kind') != 'PREPARE' or command.get('trial_id') in seen:
            raise RuntimeError('The worker requires a new PREPARE command.')
        trial_id = command['trial_id']
        seen.add(trial_id)
        context, slots = prepare(command)
        connection.send(dict(kind='READY', worker=worker, trial_id=trial_id, assigned_slots=slots))
        release = connection.recv()
        if release != dict(kind='RELEASE', trial_id=trial_id):
            raise RuntimeError('The worker requires the matching RELEASE command.')
        result = execute(context)
        connection.send(dict(kind='DONE', worker=worker, trial_id=trial_id, result=result))
