"""Freeze a separate request-registration intervention after a host-gap failure."""
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np
from .common import ROOT, SOURCE, write, digest
from .final_confirmation_jobs import clone, signature
from .final_continuous_selection import policy
from .run import save, execute
from .qualification import audit
from .decode_cap_discovery import full_native_state
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize


PLAN = 'request_registration_deferral_frozen_v1.json'


def registration_signature(cfg):
    return dict(signature(cfg), defer_D_request_registration=bool(
        cfg.get('defer_D_request_registration')))


def capsule(main):
    base = ROOT / 'references/source_capsules' / main['source_capsule']
    assert digest(base) == main['source_capsule_SHA256']
    path = base.parent / 'request_registration_deferral_overlay_v1.zip'
    assert not path.exists(), 'Do not overwrite an existing inference capsule'
    worker = 'poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    module = 'poc_flashvep/dense_ar_longbench400_pd/request_admission_deferral.py'
    before = "    channel=Client(args.address,family='AF_UNIX');engine=None;steps=[];queues=[];transfers=[];last_queue=None;sent_counts={};schedule_queues=[];allocation_refusals=[]"
    after = "    from .request_admission_deferral import wrap as wrap_registration\n" + \
        "    channel=wrap_registration(Client(args.address,family='AF_UNIX'),cfg,role);engine=None;steps=[];queues=[];transfers=[];last_queue=None;sent_counts={};schedule_queues=[];allocation_refusals=[]"
    with zipfile.ZipFile(base) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    original = files[worker]
    assert original.decode().count(before) == 1
    files[worker] = original.decode().replace(before, after).encode()
    files[module] = (SOURCE / 'request_admission_deferral.py').read_bytes()
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, files[name])
    hashes = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
    write(path.with_suffix('.json'), dict(SHA256=digest(path), files=hashes,
        base_capsule_SHA256=main['source_capsule_SHA256'],
        only_changed_existing_file=worker, added_file=module,
        original_worker_SHA256=hashlib.sha256(original).hexdigest(),
        scope='Derive from the frozen a2 source. Only the connection wrapper call and its new helper change. '
              'All original live actor files remain unchanged. Disabled policy returns the original connection. '
              'No installed runtime,model,token output or timestamp change.'))
    return path.name, digest(path)


def build():
    target = ROOT / 'configs' / PLAN
    assert not target.exists(), 'Freeze once before any new GPU measurement'
    main = json.loads((ROOT / 'configs/final_continuous_execution_plan_v1.json').read_text())
    observed = next(c for c in main['cells'] if c['workload'] == 'Heldout200'
                    and c['QPS'] == 1.8 and c['start'] == 0)
    entries = {a: json.loads((ROOT / 'configs' / (n + '.json')).read_text())
               for a, n in observed['jobs'].items()}
    native_read = next(c for a, c in entries.items() if a.startswith('Native')
                       and policy(c) == (True, True, False, None, False))
    native_completion = next(c for a, c in entries.items() if a.startswith('Native')
                             and policy(c) == (True, True, False, None, True))
    hybrid = dict(entries['BestHybrid'], defer_D_remote_setup=True)
    prototypes = dict(NativeStrong=entries['NativeStrong'], NativeRegistration=native_read,
        NativeCompletionRegistration=native_completion, LayerRegistration=entries['BestLayer'],
        TokenRegistration=entries['BestToken'], HybridRegistration=hybrid,
        TokenOldControl=entries['BestToken'])
    for label, cfg in list(prototypes.items()):
        prototypes[label] = dict(cfg, defer_D_request_registration=label not in
                                 ('NativeStrong', 'TokenOldControl'))
    source, sha = capsule(main)
    proofs = {}
    for label, prototype in prototypes.items():
        if not prototype['defer_D_request_registration']:
            continue
        gate = 'request_registration_' + label + '_gate_v1.json'
        names = []
        for count, dataset in ((32, 'continuous_qualification32_natural.json'),
                               (400, 'longbench_balanced400_v1.json')):
            name = f'RegistrationQ_{label}_Natural{count}_v1'
            cfg = clone(prototype, name, dataset, 2.8, 2026112201, (2, 3), source, sha, correctness=True)
            cfg.pop('final_composition_gate', None)
            save(cfg)
            names.append(name)
        proofs[label] = dict(jobs=names, gate=gate, signature=registration_signature(prototype))
    cells = []
    jobs = []
    pilot_arms = ('NativeCompletionRegistration', 'TokenRegistration', 'TokenOldControl')
    designs = [('OBSERVED_TRACE_DIAGNOSTIC', 'balanced400_heldout.json', 2026112411, 0, pilot_arms)]
    designs += [('INDEPENDENT_CONFIRMATION', 'longbench_balanced400_v1.json', seed, index, tuple(prototypes))
                for index, seed in enumerate((2026112771, 2026112772, 2026112773))]
    for stage, dataset, seed, index, labels in designs:
        arms = {}
        for label in labels:
            prefix = 'RegistrationPilot_Heldout200' if stage == 'OBSERVED_TRACE_DIAGNOSTIC' else 'RegistrationConfirm_Main400'
            name = f'{prefix}_q1.8_s{index}_{label}_v1'
            cfg = clone(prototypes[label], name, dataset, 1.8, seed, (2, 3), source, sha)
            cfg.pop('final_composition_gate', None)
            if label in proofs:
                cfg['registration_deferral_gate'] = proofs[label]['gate']
            cfg.update(registration_intervention=True, scope='Adaptive registration-only follow-up. '
                'All new seeds and seven arms are fixed before the pilot. Original main140 remains unchanged. '
                'Run independent confirmation regardless of pilot direction,subject to correctness. '
                'Native policy controls and strong compiled Token remain. Normal completion only.')
            save(cfg)
            arms[label] = name
        ordered = list(arms.values())
        np.random.default_rng(seed + 19).shuffle(ordered)
        jobs.extend(ordered)
        cells.append(dict(stage=stage, dataset=dataset, QPS=1.8, seed=seed,
                          start=index, GPU_pair=[2, 3], jobs=arms))
    write(target, dict(time=time.time(), source_capsule=source, source_capsule_SHA256=sha,
        original_main_plan_SHA256=digest(ROOT / 'configs/final_continuous_execution_plan_v1.json'),
        symptom='Heldout200 q1.8 s0 strong Token: first-output inter-step host gaps6.778/11.726/8.262/3.874ms '
                'for requests195/170/122/108. Other D registrations complete inside all four gaps. '
                'The corresponding Native gaps are0.257/0.247/0.208/0.220ms. No JIT event or recorded READ overlap.',
        hypotheses=[
            'H1: New request registration and input processing delay the next decode submission. '
            'Deferring registration should reduce first-transition host gaps and TPOT tails.',
            'H2: IPC receive/unpickling dominates the gap. Registration deferral retains IPC reads,so large gaps should persist.',
            'H3: First-decode GPU/graph preparation or natural batch variation dominates. '
            'Registration-only intervention should not consistently improve the independent paired metric tails.'],
        qualification=proofs, cells=cells, performance_jobs=jobs,
        evidence='ADAPTIVE_HOST_ADMISSION_INTERVENTION_FROZEN_BEFORE_NEW_TIMINGS',
        scope='Ten Natural32/400 proof starts and24 serving starts. All use2/3 after its original main lane drains. '
              'Only request registration is delayed within each action. The shared READ/compute/completion policies remain explicit. '
              'No active-decode prefill insertion,no future EOS knowledge,no changed SLO,no modified timestamp or output. '
              'This policy also changes request admission timing and subsequent batching. It is not exclusive Python-cost attribution. '
              'The pilot repeats an observed request trace on a different pair. It has no independent-start CI. '
              'Confirmation uses three new arrival seeds on the same400 requests,not globally new request generalization.'))
    print('REGISTRATION_INTERVENTION_FROZEN', len(proofs) * 2, 'proofs', len(jobs), 'serving starts', flush=True)


def run_plan(plan, prefix='request_registration_deferral'):
    for label, item in plan['qualification'].items():
        path = ROOT / 'state' / item['gate']
        if path.exists():
            continue
        results = []
        try:
            for name in item['jobs']:
                assert execute(name), name
                cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
                state = audit(name) if cfg.get('axis_kind') else full_native_state(name, False)
                assert state['passed']
                if cfg.get('axis_kind'):
                    assert state['decode_full36_observation_complete']
                    assert all(v['status'] == 'PASS' for v in state['imported_prefix_KV'])
                score = quality(name)
                assert score['passed']
                registration = json.loads((ROOT / 'raw' / name / 'D_request_registration_deferral.json').read_text())
                assert registration['passed']
                results.append(dict(name=name, state=state, quality=score,
                                    compute_rule=no_mixed(name)))
            proof = dict(passed=True, full400_passed=True, results=results,
                         signature=item['signature'], source_capsule_SHA256=plan['source_capsule_SHA256'],
                         evidence='CORRECTNESS_ONLY')
        except Exception as error:
            proof = dict(passed=False, full400_passed=False, results=results, error=repr(error),
                         signature=item['signature'], source_capsule_SHA256=plan['source_capsule_SHA256'],
                         evidence='FAILED')
        write(path, proof)
        print('REGISTRATION_PROOF', label, proof['passed'], flush=True)
    results = []
    for name in plan['performance_jobs']:
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        gate = cfg.get('registration_deferral_gate')
        if gate and not json.loads((ROOT / 'state' / gate).read_text())['passed']:
            result = dict(name=name, status='NOT_RUN', reason='BLOCKED_CORRECTNESS', gate=gate)
        elif execute(name):
            state = audit(name) if cfg.get('axis_kind') else None
            if state:
                assert state['passed'] and state['decode_full36_observation_complete']
            score = quality(name)
            m = summarize(name)
            result = dict(name=name, status='MEASURED_ONLINE' if score['passed'] else 'FAILED_TASK_QUALITY',
                          quality=score, **{f:m[f] for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')})
        else:
            result = dict(name=name, status='FAILED')
        results.append(result)
        write(ROOT / 'state' / (prefix+'_progress_v1.json'), dict(results=results))
        print('REGISTRATION_SERVING', result, flush=True)
    write(ROOT / 'state' / (prefix+'_complete_v1.json'),
          dict(complete=True, results=results, scope='Operational completion,not safety or performance GO.'))


def run():
    plan = json.loads((ROOT / 'configs' / PLAN).read_text())
    assert (ROOT / 'state/final_performance_lane1_complete_v1.json').exists()
    run_plan(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    args = parser.parse_args()
    build() if args.action == 'build' else run()
