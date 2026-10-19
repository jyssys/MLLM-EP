"""CPU-only assignment, barrier, metrics, freeze, and child-cleanup regressions."""
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read
from poc_ouro_parallelism_deep_discovery.dp4_cohort import (
    aggregate, assignments, collect_messages, run_cohort, serve_worker)
from poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_control import cleanup, freeze_plan


def case(count=8, context=2):
    return dict(C=count, context=context, requests=[dict(cohort_slot=index,
        source_request_id=f'fixture{index % 2}', prompt_token_ids=[7] * context) for index in range(count)])


def result(worker, requests, length=128):
    return dict(worker=worker, assigned_slots=[r['cohort_slot'] for r in requests],
        generated_tokens=len(requests) * length, local_generate_wall_ns=10 if requests else 0,
        request_records=[dict(cohort_slot=r['cohort_slot'], output_tokens=length) for r in requests],
        output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS')


class DP4Tests(unittest.TestCase):
    def test_round_robin_and_idle_workers(self):
        expected = {1: [[0], [], [], []], 2: [[0], [1], [], []],
                    4: [[0], [1], [2], [3]], 8: [[0, 4], [1, 5], [2, 6], [3, 7]]}
        for count, slots in expected.items():
            self.assertEqual([[r['cohort_slot'] for r in group] for group in assignments(case(count))], slots)

    def test_assignment_rejects_duplicates_and_wrong_context(self):
        value = case(2)
        value['requests'][1]['cohort_slot'] = 0
        with self.assertRaises(ValueError):
            assignments(value)
        value = case(2)
        value['requests'][0]['prompt_token_ids'].append(8)
        with self.assertRaises(ValueError):
            assignments(value)

    def test_parent_makespan_is_not_max_or_sum_worker_wall(self):
        value = case()
        results = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
        measured = aggregate(value, results, 128, 1000, 1200)
        self.assertEqual(measured['global_makespan_ns'], 200)
        self.assertEqual(measured['generated_tokens'], 1024)
        self.assertEqual(measured['cohort_output_TPS'], 1024 * 1e9 / 200)
        self.assertEqual(measured['reserved_replicas'], 4)

    def test_aggregate_rejects_denominator_or_missing_worker(self):
        value = case()
        results = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
        results[1]['generated_tokens'] -= 1
        with self.assertRaises(ValueError):
            aggregate(value, results, 128, 0, 100)
        del results[1]
        with self.assertRaises(ValueError):
            aggregate(value, results, 128, 0, 100)

    def test_aggregate_rejects_wrong_slots_or_idle_generate(self):
        value = case(1)
        results = {worker: result(worker, group) for worker, group in enumerate(assignments(value))}
        self.assertEqual(aggregate(value, results, 128, 0, 100)['active_replicas'], 1)
        results[2]['local_generate_wall_ns'] = 1
        with self.assertRaises(ValueError):
            aggregate(value, results, 128, 0, 100)
        results[2]['local_generate_wall_ns'] = 0
        results[0]['assigned_slots'] = [1]
        with self.assertRaises(ValueError):
            aggregate(value, results, 128, 0, 100)

    def test_collect_rejects_wrong_identity_kind_and_trial(self):
        for message in [dict(kind='READY', worker=1, trial_id=7),
                        dict(kind='ERROR', worker=0, failure='FIXTURE_FAILURE'),
                        dict(kind='READY', worker=0, trial_id=6)]:
            parent, child = Pipe()
            try:
                child.send(message)
                with self.assertRaises(RuntimeError):
                    collect_messages({0: parent}, 'READY', 1, trial_id=7)
            finally:
                parent.close()
                child.close()

    def test_collect_deadline_and_eof(self):
        parent, child = Pipe()
        try:
            with self.assertRaises(TimeoutError):
                collect_messages({0: parent}, 'READY', 0.001)
            child.close()
            with self.assertRaises(RuntimeError):
                collect_messages({0: parent}, 'READY', 1)
        finally:
            parent.close()
            child.close()

    def test_worker_executes_only_after_matching_release(self):
        parent, child = Pipe()
        executed = threading.Event()
        errors = []
        def target():
            try:
                serve_worker(child, 0, lambda command: (command, [0]),
                             lambda context: (executed.set(), dict(value=1))[1])
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            parent.send(dict(kind='PREPARE', trial_id=3))
            self.assertTrue(parent.poll(1))
            self.assertEqual(parent.recv()['kind'], 'READY')
            self.assertFalse(executed.is_set())
            parent.send(dict(kind='RELEASE', trial_id=3))
            self.assertTrue(parent.poll(1))
            self.assertEqual(parent.recv()['kind'], 'DONE')
            self.assertTrue(executed.is_set())
            parent.send(dict(kind='STOP'))
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
        finally:
            parent.close()
            child.close()

    def test_worker_rejects_stale_release_without_execution(self):
        parent, child = Pipe()
        errors, executed = [], []
        def target():
            try:
                serve_worker(child, 0, lambda command: (command, [0]), lambda context: executed.append(1))
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            parent.send(dict(kind='PREPARE', trial_id=3))
            self.assertTrue(parent.poll(1))
            parent.recv()
            parent.send(dict(kind='RELEASE', trial_id=2))
            thread.join(1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertEqual(executed, [])
        finally:
            parent.close()
            child.close()

    def test_all_four_ready_before_any_execute_and_reusable_workers(self):
        value = case(2)
        connections, child_ends, threads, prepared, errors = {}, [], [], [], []
        lock = threading.Lock()
        def start(worker):
            parent, child = Pipe()
            connections[worker] = parent
            child_ends.append(child)
            def prepare(command):
                with lock:
                    prepared.append((command['trial_id'], worker))
                group = assignments(value)[worker]
                return (command, group), [r['cohort_slot'] for r in group]
            def execute(context):
                command, group = context
                self.assertEqual(sum(trial == command['trial_id'] for trial, _ in prepared), 4)
                return result(worker, group, command['length'])
            def target():
                try:
                    serve_worker(child, worker, prepare, execute)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            threads.append(thread)
        for worker in range(4):
            start(worker)
        try:
            for trial in range(2):
                record = run_cohort(connections, value, 0, 128, 'CLEAN', trial, 2, [])
                self.assertEqual(record['generated_tokens'], 256)
                self.assertEqual(record['active_replicas'], 2)
            for connection in connections.values():
                connection.send(dict(kind='STOP'))
            for thread in threads:
                thread.join(1)
                self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
        finally:
            for connection in list(connections.values()) + child_ends:
                connection.close()

    def test_freeze_actual_cpu_metadata_for_both_sizes(self):
        protocol = ROOT / 'artifacts/native_dp4_cohort_protocol_v1.json'
        for size in ('1.4B', '2.6B'):
            with tempfile.TemporaryDirectory(prefix='ouro_dp4_cpu_') as temporary:
                plan = freeze_plan(size, protocol, Path(temporary))
                self.assertEqual(plan['status'], 'FROZEN_BEFORE_WORKER_START')
                self.assertEqual([w['physical_gpu'] for w in plan['workers']], [4, 5, 6, 7])
                self.assertEqual(len(plan['cases']), 8)
                for worker in plan['workers']:
                    config = worker['engine_config']
                    self.assertEqual(config['tensor_parallel_size'], 1)
                    self.assertEqual(config['pipeline_parallel_size'], 1)
                    self.assertEqual(config['data_parallel_size'], 1)
                    self.assertEqual(config['kv_cache_memory_bytes'], 52 * 1024 ** 3)
                    self.assertTrue(config['async_scheduling'])
                    self.assertTrue(all(Path(root).is_relative_to(worker['output'])
                                        for root in worker['cache_environment'].values()))
                frozen = read(Path(temporary) / 'dp4_plan.json')
                self.assertEqual(frozen['fixed_token_id'], plan['fixed_token_id'])

    def test_known_child_cleanup_on_timeout(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        try:
            records = cleanup({0: process}, {}, 0)
            self.assertEqual(records[0]['pid'], process.pid)
            self.assertTrue(records[0]['forced'])
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_subprocess_fd_singleton_mask_and_graceful_cleanup(self):
        parent, child = Pipe()
        command = [sys.executable, '-c',
            "import os,sys; from multiprocessing.connection import Connection; "
            "c=Connection(int(sys.argv[1])); "
            "c.send(dict(kind='LOADED',worker=0,mask=os.environ['CUDA_VISIBLE_DEVICES'],"
            "guard_output=os.environ['LOOPED_KV_OUTPUT'],pgid=os.getpgid(0))); "
            "assert c.recv()==dict(kind='STOP'); c.close()", str(child.fileno())]
        process = subprocess.Popen(command, pass_fds=(child.fileno(),),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='6', LOOPED_KV_OUTPUT='CPU_GUARD_FIXTURE'))
        child.close()
        try:
            loaded = collect_messages({0: parent}, 'LOADED', 5)[0]
            self.assertEqual(loaded['mask'], '6')
            self.assertEqual(loaded['guard_output'], 'CPU_GUARD_FIXTURE')
            self.assertEqual(loaded['pgid'], os.getpgid(0))
            records = cleanup({0: process}, {0: parent}, 5)
            self.assertFalse(records[0]['forced'])
            self.assertEqual(records[0]['exit_code'], 0)
        finally:
            parent.close()
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_entrypoints_help_does_not_import_torch(self):
        for name in ('native_dp4_cohort_control', 'native_dp4_cohort_worker'):
            command = [sys.executable, '-c',
                f"import sys; import poc_ouro_parallelism_deep_discovery.scripts.{name}; assert 'torch' not in sys.modules"]
            subprocess.run(command, check=True, capture_output=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
