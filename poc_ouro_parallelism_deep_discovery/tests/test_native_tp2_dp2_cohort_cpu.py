"""CPU-only TP2xDP2 placement, message, ownership, freeze, and timing checks."""
import copy
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, UUIDS
from poc_ouro_parallelism_deep_discovery.dp4_cohort import collect_messages, serve_worker
from poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_control import cleanup
from poc_ouro_parallelism_deep_discovery.scripts.native_tp2_dp2_cohort_control import freeze_plan, frontend_environment
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import (
    GROUPS, aggregate, assignments, inspect_process_leases, run_cohort, validate_groups, validate_ownership)


def case(count=8, context=2):
    return dict(C=count, context=context, requests=[dict(cohort_slot=index,
        source_request_id=f'fixture{index % 2}', prompt_token_ids=[7] * context) for index in range(count)])


def result(worker, requests, length=128, clock_base=1010):
    return dict(worker=worker, physical_gpus=list(GROUPS[worker]),
        assigned_slots=[r['cohort_slot'] for r in requests], generated_tokens=len(requests) * length,
        local_generate_wall_ns=10 if requests else 0,
        release_received_ns=clock_base,
        local_generate_start_ns=clock_base + 10 if requests else None,
        local_generate_end_ns=clock_base + 20 if requests else None,
        request_records=[dict(cohort_slot=r['cohort_slot'], source_request_id=r['source_request_id'],
                              output_tokens=length, finish_reason='length') for r in requests],
        output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS')


def ownership(group=GROUPS[0], layers=24):
    return [dict(rank=rank, world=2, pid=100 + rank, device=f'cuda:{rank}',
        GPU_uuid=UUIDS[physical - 4], CUDA_VISIBLE_DEVICES=','.join(map(str, group)),
        physical_layers=layers, cache_slots=4 * layers,
        geometry=[dict(Q_heads=8, KV_heads=8, head_dim=128, loop_cache_modules=4) for _ in range(layers)])
        for rank, physical in enumerate(group)]


class TP2DP2Tests(unittest.TestCase):
    def test_frontend_title_option_preserves_strict_guard_environment(self):
        specification = dict(worker=1, physical_gpus=[6, 7],
            cache_environment={'VLLM_CACHE_ROOT': '/fixture/cache'})
        original = dict(os.environ)
        with patch.dict(os.environ, dict(LOOPED_KV_OUTPUT='/fixture/guard', RANK='99',
                                        SPT_NOENV='', VLLM_ALLOW_INSECURE_SERIALIZATION='0')):
            environment = frontend_environment(specification)
            self.assertEqual(environment['SPT_NOENV'], '1')
            self.assertEqual(environment['LOOPED_KV_OUTPUT'], '/fixture/guard')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '6,7')
            self.assertEqual(environment['VLLM_WORKER_MULTIPROC_METHOD'], 'spawn')
            self.assertEqual(environment['VLLM_ALLOW_INSECURE_SERIALIZATION'], '0')
            self.assertEqual(environment['VLLM_CACHE_ROOT'], '/fixture/cache')
            self.assertNotIn('RANK', environment)
            self.assertEqual(os.environ['SPT_NOENV'], '')
        self.assertEqual(dict(os.environ), original)

    def test_all_frozen_counts_cover_slots_once(self):
        expected = {1: [[0], []], 2: [[0], [1]], 4: [[0, 2], [1, 3]],
                    8: [[0, 2, 4, 6], [1, 3, 5, 7]]}
        for count, slots in expected.items():
            actual = assignments(case(count))
            self.assertEqual([[r['cohort_slot'] for r in group] for group in actual], slots)
            self.assertEqual(sorted(r['cohort_slot'] for group in actual for r in group), list(range(count)))

    def test_invalid_physical_membership_rejected(self):
        self.assertTrue(validate_groups([[4, 5], [6, 7]]))
        for groups in [[[4, 6], [5, 7]], [[4, 5], [5, 7]], [[4, 5]], [[0, 1], [2, 3]]]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                validate_groups(groups)

    def test_assignment_rejects_count_duplicate_and_context(self):
        for mutate in [lambda c: c.update(C=3), lambda c: c['requests'][1].update(cohort_slot=0),
                       lambda c: c['requests'][0]['prompt_token_ids'].append(1)]:
            value = case(2)
            mutate(value)
            with self.assertRaises(ValueError):
                assignments(value)

    def test_parent_wall_is_observed_not_worker_max(self):
        value = case(8)
        rows = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
        measured = aggregate(value, rows, 128, 1000, 1300)
        self.assertEqual(measured['global_makespan_ns'], 300)
        self.assertEqual(measured['generated_tokens'], 1024)
        self.assertEqual(measured['cohort_output_TPS'], 1024 * 1e9 / 300)
        self.assertEqual(measured['reserved_gpus'], 4)
        self.assertEqual(measured['reserved_replicas'], 2)

    def test_idle_group_does_not_generate(self):
        value = case(1)
        rows = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
        self.assertEqual(aggregate(value, rows, 128, 1000, 1300)['active_replicas'], 1)
        rows[1]['local_generate_wall_ns'] = 1
        with self.assertRaises(ValueError):
            aggregate(value, rows, 128, 1000, 1300)

    def test_aggregate_rejects_denominator_group_and_slots(self):
        value = case(4)
        for mutation in [lambda r: r[0].update(generated_tokens=1),
                         lambda r: r[0].update(physical_gpus=[4, 6]),
                         lambda r: r[0].update(assigned_slots=[0, 1]),
                         lambda r: r[0]['request_records'][0].update(output_tokens=127),
                         lambda r: r[0]['request_records'][0].update(source_request_id='WRONG_SOURCE'),
                         lambda r: r[0]['request_records'][0].update(finish_reason='eos'),
                         lambda r: r[0].update(output_identity='NOT_CHECKED'),
                         lambda r: r[0].update(local_generate_wall_ns=1000),
                         lambda r: r.pop(1)]:
            rows = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
            mutation(rows)
            with self.assertRaises(ValueError):
                aggregate(value, rows, 128, 1000, 1300)

    def test_release_and_active_endpoints_are_enclosed(self):
        value = case(2)
        for mutation in [lambda r: r[0].update(release_received_ns=999),
                         lambda r: r[0].update(release_received_ns=1301),
                         lambda r: r[0].update(local_generate_start_ns=1009),
                         lambda r: r[0].update(local_generate_end_ns=1301),
                         lambda r: r[0].update(local_generate_start_ns=1040),
                         lambda r: r[0].update(local_generate_wall_ns=9),
                         lambda r: r[0].update(local_generate_end_ns=None)]:
            rows = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
            mutation(rows)
            with self.assertRaises(ValueError):
                aggregate(value, rows, 128, 1000, 1300)

    def test_idle_endpoints_must_both_be_none(self):
        for field in ['local_generate_start_ns', 'local_generate_end_ns']:
            value = case(1)
            rows = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
            rows[1][field] = 1020
            with self.assertRaises(ValueError):
                aggregate(value, rows, 128, 1000, 1300)

    def test_message_identity_trial_and_error_guards(self):
        for message in [dict(kind='READY', worker=1, trial_id=5),
                        dict(kind='READY', worker=0, trial_id=4),
                        dict(kind='ERROR', worker=0, trial_id=5)]:
            parent, child = Pipe()
            try:
                child.send(message)
                with self.assertRaises(RuntimeError):
                    collect_messages({0: parent}, 'READY', 1, trial_id=5)
            finally:
                parent.close()
                child.close()

    def test_worker_rejects_wrong_release(self):
        parent, child = Pipe()
        errors, executions = [], []
        def target():
            try:
                serve_worker(child, 0, lambda command: (command, [0]), lambda context: executions.append(1))
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            parent.send(dict(kind='PREPARE', trial_id=5))
            self.assertTrue(parent.poll(1))
            self.assertEqual(parent.recv()['kind'], 'READY')
            parent.send(dict(kind='RELEASE', trial_id=4))
            thread.join(1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(executions, [])
        finally:
            parent.close()
            child.close()

    def test_timing_begins_after_ready_and_includes_release_sends(self):
        value = case(2)
        events = []
        class Sender:
            def __init__(self, worker):
                self.worker = worker
            def send(self, message):
                events.append((message['kind'], self.worker))
        connections = {worker: Sender(worker) for worker in range(2)}
        def collect(connections, kind, timeout, trial_id=None, transcript=None):
            events.append((kind, 'COLLECT'))
            if kind == 'READY':
                return {worker: dict(assigned_slots=[worker]) for worker in range(2)}
            return {worker: dict(result=result(worker, assignments(value)[worker])) for worker in range(2)}
        def clock():
            events.append(('CLOCK', len(events)))
            return 1000 if sum(e[0] == 'CLOCK' for e in events) == 1 else 1400
        with patch('poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort.collect_messages', side_effect=collect), \
             patch('poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort.time.perf_counter_ns', side_effect=clock):
            observed = run_cohort(connections, value, 0, 128, 'CLEAN', 5, 1, [])
        self.assertEqual([event[0] for event in events],
                         ['PREPARE', 'PREPARE', 'READY', 'CLOCK', 'RELEASE', 'RELEASE', 'DONE', 'CLOCK'])
        self.assertEqual(observed['global_makespan_ns'], 400)

    def test_both_groups_ready_and_reusable(self):
        value = case(1)
        connections, children, threads, prepared, errors = {}, [], [], [], []
        lock = threading.Lock()
        for worker in range(2):
            parent, child = Pipe()
            connections[worker] = parent
            children.append(child)
            def target(worker=worker, child=child):
                def prepare(command):
                    with lock:
                        prepared.append((command['trial_id'], worker))
                    group = assignments(value)[worker]
                    return (command, group), [r['cohort_slot'] for r in group]
                def execute(context):
                    command, group = context
                    self.assertEqual(sum(trial == command['trial_id'] for trial, _ in prepared), 2)
                    return result(worker, group, command['length'], clock_base=time.perf_counter_ns())
                try:
                    serve_worker(child, worker, prepare, execute)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            threads.append(thread)
        try:
            for trial in range(2):
                observed = run_cohort(connections, value, 0, 128, 'CLEAN', trial, 2, [])
                self.assertEqual(observed['generated_tokens'], 128)
            for connection in connections.values():
                connection.send(dict(kind='STOP'))
            for thread in threads:
                thread.join(1)
                self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
        finally:
            for connection in list(connections.values()) + children:
                connection.close()

    def test_run_rejects_missing_frontend(self):
        with self.assertRaises(ValueError):
            run_cohort({}, case(), 0, 128, 'CLEAN', 0, 1, [])

    def test_tp2_UUID_coverage_for_both_groups_and_sizes(self):
        for group in GROUPS:
            for layers in [24, 48]:
                self.assertTrue(validate_ownership(group, layers, ownership(group, layers)))

    def test_tp2_wrong_rank_UUID_world_and_geometry_rejected(self):
        for mutate in [lambda rows: rows[1].update(rank=0),
                       lambda rows: rows[1].update(GPU_uuid=rows[0]['GPU_uuid']),
                       lambda rows: rows[1].update(world=1),
                       lambda rows: rows[1].update(CUDA_VISIBLE_DEVICES='4,6'),
                       lambda rows: rows[1].update(cache_slots=4),
                       lambda rows: rows[1].update(pid=rows[0]['pid']),
                       lambda rows: rows[1]['geometry'][0].update(KV_heads=16)]:
            rows = ownership()
            mutate(rows)
            with self.assertRaises(ValueError):
                validate_ownership(GROUPS[0], 24, rows)

    def test_freeze_metadata_for_both_sizes(self):
        for size in ['1.4B', '2.6B']:
            with tempfile.TemporaryDirectory(prefix='ouro_tp2_dp2_cpu_') as directory:
                plan = freeze_plan(size, ROOT / 'artifacts/native_tp2_dp2_cohort_protocol_v1.json', Path(directory))
                self.assertEqual(len(plan['cases']), 8)
                self.assertEqual(len(plan['workers']), 2)
                self.assertTrue(plan['exact_tp1_tp2_inputs_equal'])
                self.assertTrue(plan['exact_templates_equal_except_tp'])
                self.assertEqual(plan['native_protocol_sha256'], '240eb229b37734a532c38c7e08d008a3dc2cabb98ec3c7be7d0f64d46b7d42f9')
                for worker in plan['workers']:
                    config = worker['engine_config']
                    self.assertEqual(config['tensor_parallel_size'], 2)
                    self.assertEqual(config['data_parallel_size'], 1)
                    self.assertEqual(config['pipeline_parallel_size'], 1)
                    self.assertEqual(config['kv_cache_memory_bytes'], 52 * 1024 ** 3)
                    self.assertTrue(all(Path(root).is_relative_to(worker['output']) for root in worker['cache_environment'].values()))
                self.assertEqual(read(Path(directory) / 'tp2_dp2_plan.json')['fixed_token_id'], 378)

    def test_wrong_protocol_membership_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory(prefix='ouro_tp2_dp2_protocol_') as directory:
            protocol = read(ROOT / 'artifacts/native_tp2_dp2_cohort_protocol_v1.json')
            protocol['physical_groups'] = [[4, 6], [5, 7]]
            path = Path(directory) / 'wrong.json'
            # Generated CPU fixture only. No existing protocol changes.
            import json
            path.write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                freeze_plan('1.4B', path, Path(directory) / 'output')
            self.assertFalse((Path(directory) / 'output').exists())

    def test_real_CPU_subprocess_mask_FD_and_session_lease(self):
        parent, child = Pipe()
        script = ("import os,sys; from multiprocessing.connection import Connection; "
                  "c=Connection(int(sys.argv[1])); c.send(dict(kind='LOADED',worker=0,pid=os.getpid(),"
                  "mask=os.environ['CUDA_VISIBLE_DEVICES'])); assert c.recv()==dict(kind='STOP'); c.close()")
        process = subprocess.Popen([sys.executable, '-c', script, str(child.fileno())],
            pass_fds=(child.fileno(),), env=dict(os.environ, CUDA_VISIBLE_DEVICES='4,5', LOOPED_KV_OUTPUT='CPU_TP2_DP2_FIXTURE'))
        child.close()
        try:
            loaded = collect_messages({0: parent}, 'LOADED', 5)[0]
            self.assertEqual(loaded['mask'], '4,5')
            self.assertEqual(loaded['pid'], process.pid)
            self.assertEqual(len(inspect_process_leases([process.pid], os.getpgid(0), os.getsid(0), 'CPU_TP2_DP2_FIXTURE')), 1)
            with self.assertRaises(RuntimeError):
                inspect_process_leases([process.pid], os.getpgid(0), os.getsid(0), 'WRONG_OUTPUT')
            records = cleanup({0: process}, {0: parent}, 5)
            self.assertFalse(records[0]['forced'])
            self.assertEqual(records[0]['exit_code'], 0)
        finally:
            parent.close()
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_entrypoint_imports_do_not_import_torch(self):
        for module in ['native_tp2_dp2_cohort_control', 'native_tp2_dp2_cohort_worker']:
            command = [sys.executable, '-c', f'import sys; import poc_ouro_parallelism_deep_discovery.scripts.{module}; assert "torch" not in sys.modules']
            subprocess.run(command, check=True, capture_output=True, timeout=10)

    def test_real_CPU_grandchild_ancestry_is_not_direct_child_only(self):
        parent, child = Pipe()
        middle = ("import sys,subprocess; "
                  "p=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.buffer.read(1)'],stdin=subprocess.PIPE); "
                  "print(p.pid,flush=True); sys.stdin.buffer.read(1); "
                  "p.stdin.write(b'x'); p.stdin.flush(); p.stdin.close(); p.wait(timeout=5)")
        script = ("import os,sys,subprocess; from multiprocessing.connection import Connection; "
                  "c=Connection(int(sys.argv[1])); "
                  f"p=subprocess.Popen([sys.executable,'-c',{middle!r}],stdin=subprocess.PIPE,stdout=subprocess.PIPE); "
                  "rank_pid=int(p.stdout.readline()); "
                  "c.send(dict(kind='LOADED',worker=0,pid=os.getpid(),middle_pid=p.pid,rank_pid=rank_pid)); "
                  "assert c.recv()==dict(kind='STOP'); p.stdin.write(b'x'); p.stdin.flush(); "
                  "p.stdin.close(); p.wait(timeout=5); c.close()")
        process = subprocess.Popen([sys.executable, '-c', script, str(child.fileno())],
            pass_fds=(child.fileno(),), env=dict(os.environ, LOOPED_KV_OUTPUT='CPU_TP2_DESCENDANTS'))
        child.close()
        try:
            loaded = collect_messages({0: parent}, 'LOADED', 5)[0]
            leases = inspect_process_leases([loaded['rank_pid']], os.getpgid(0), os.getsid(0),
                'CPU_TP2_DESCENDANTS', ancestor_pid=process.pid)
            self.assertEqual(leases[0]['ancestry_to_frontend'],
                             [loaded['rank_pid'], loaded['middle_pid'], process.pid])
            with self.assertRaises(RuntimeError):
                inspect_process_leases([process.pid], os.getpgid(0), os.getsid(0),
                    'CPU_TP2_DESCENDANTS', ancestor_pid=loaded['rank_pid'])
            records = cleanup({0: process}, {0: parent}, 5)
            self.assertFalse(records[0]['forced'])
            self.assertEqual(records[0]['exit_code'], 0)
        finally:
            parent.close()
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
