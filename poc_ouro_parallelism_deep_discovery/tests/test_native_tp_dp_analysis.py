import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from poc_ouro_parallelism_deep_discovery.scripts.analyze_native_tp_dp_cohorts import (
    Inputs, analyze, availability, trial_metrics)


def fixture():
    case = dict(C=2,context=2,requests=[dict(cohort_slot=i,prompt_token_ids=[7,8],source_request_id=f'p{i}') for i in range(2)])
    raw = dict(size='1.4B',R=4,C=2,context=2,generated_tokens=256,quality='NOT_RUN',
        DP=4,TP=1,lane='CLEAN',reserved_replicas=4,active_replicas=2,physical_gpus=[4,5,6,7],
        release_start_ns=1000,all_completed_ns=2001000,global_makespan_ns=2000000,
        global_makespan_ms=2,cohort_output_TPS=128000,workers=[])
    for worker in range(4):
        slots = [worker] if worker<2 else []
        raw['workers'].append(dict(worker=worker,physical_gpu=worker+4,assigned_slots=slots,
            generated_tokens=len(slots)*128,output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS',
            local_generate_wall_ns=1000000 if slots else 0,release_received_ns=1100+worker,
            schedule_trace=None,request_records=[dict(cohort_slot=i,source_request_id=f'p{i}',
                output_tokens=128,finish_reason='length') for i in slots]))
    return case,raw


class AnalysisTests(unittest.TestCase):
    def test_incomplete_jobs_do_not_parse_partial_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'artifacts').mkdir()
            protocol=dict(cohort_sizes=[1,2,4,8],contexts=[1024,4096],generated_tokens=128,
                          repeats=5,depth_mode='FIXED_R4_ONLY')
            encoded=json.dumps(protocol).encode()
            (root/'artifacts/native_cohort_protocol_v2.json').write_bytes(encoded)
            (root/'artifacts/native_dp4_cohort_protocol_v1.json').write_text(
                json.dumps(dict(base_protocol_sha256=hashlib.sha256(encoded).hexdigest())))
            job=root/'outputs/native_cohort_base14_P1_v1'
            job.mkdir(parents=True)
            (job/'status.json').write_text(json.dumps(dict(success=False)))
            (job/'guard_status.json').write_text(json.dumps(dict(success=True,exit_code=0)))
            (job/'C1_N1024_repeat0.json').write_text('not complete JSON')
            value=analyze(root)
            self.assertEqual(value['trials'],[])
            self.assertTrue(all(r['status']=='NOT_AVAILABLE' for r in value['summaries']))

    def test_requires_runner_and_guard_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'status.json').write_text(json.dumps(dict(success=True)))
            self.assertEqual(availability(root,Inputs(root))['status'],'NOT_AVAILABLE')
            (root/'guard_status.json').write_text(json.dumps(dict(success=False,exit_code=0)))
            self.assertEqual(availability(root,Inputs(root))['status'],'NOT_AVAILABLE')
            (root/'guard_status.json').write_text(json.dumps(dict(success=True,exit_code=0)))
            self.assertEqual(availability(root,Inputs(root))['status'],'AVAILABLE')

    def test_guard_exit_code_and_partial_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'status.json').write_text('{')
            (root/'guard_status.json').write_text(json.dumps(dict(success=True,exit_code=1)))
            self.assertEqual(availability(root,Inputs(root))['status'],'NOT_AVAILABLE')

    def test_parent_interval_is_not_inferred_from_worker_max(self):
        case,raw=fixture()
        metric,workers=trial_metrics(raw,'DP4','1.4B',case,128)
        self.assertEqual(metric['wall_ms'],2)
        self.assertEqual(metric['longest_active_local_generate_ms'],1)
        self.assertEqual(metric['charged_IPC_controller_other_residual_ms'],1)
        self.assertEqual(sum(w['active'] for w in workers),2)

    def test_parent_timestamp_mismatch_rejected(self):
        case,raw=fixture(); raw['global_makespan_ns']+=1
        with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_wrong_assignment_rejected(self):
        case,raw=fixture(); raw['workers'][1]['assigned_slots']=[0]
        with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_wrong_prompt_source_rejected(self):
        case,raw=fixture(); raw['workers'][1]['request_records'][0]['source_request_id']='other'
        with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_trace_contamination_rejected(self):
        case,raw=fixture(); raw['workers'][0]['schedule_trace']=[]
        with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_idle_generate_and_worker_outside_parent_rejected(self):
        case,original=fixture()
        for worker,local in [(2,1),(0,3000000)]:
            raw=copy.deepcopy(original); raw['workers'][worker]['local_generate_wall_ns']=local
            with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_output_denominator_rejected(self):
        case,raw=fixture(); raw['workers'][0]['request_records'][0]['output_tokens']=127
        with self.assertRaises(ValueError): trial_metrics(raw,'DP4','1.4B',case,128)

    def test_tp_api_boundary(self):
        case,_=fixture()
        raw=dict(size='1.4B',R=4,C=2,context=2,generated_tokens=256,quality='NOT_RUN',tp=4,
            output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS',wall_ms=4,cohort_output_TPS=64000,
            request_records=[dict(output_tokens=128,finish_reason='length') for _ in range(2)])
        metric,workers=trial_metrics(raw,'TP4','1.4B',case,128)
        self.assertEqual(metric['wall_ms'],4)
        self.assertIsNone(metric['charged_IPC_controller_other_residual_ms'])
        self.assertEqual(workers,[])
        raw['cohort_output_TPS']*=4
        with self.assertRaises(ValueError): trial_metrics(raw,'TP4','1.4B',case,128)


if __name__=='__main__':
    unittest.main()
