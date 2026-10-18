"""Freeze continuity, sources, and workload rules before new inference."""
import hashlib
import subprocess
from datetime import datetime, timezone
from poc_decode_first_depth_aware_dp_v3.contract import *

def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True)

def main():
    target = ROOT / 'artifacts/continuity.json'
    assert not target.exists(), 'The continuity snapshot is immutable.'
    original = '/home/esjung/MLLM-EP-github'
    git = lambda *args: command(['git', '-C', original, *args])
    snapshot = dict(created_utc=datetime.now(timezone.utc).isoformat(), base_commit=BASE_COMMIT,
        specs={str(p): sha(p) for p in SPECS}, original_head=git('rev-parse', 'HEAD').strip(),
        original_staged_sha256=hashlib.sha256(git('diff','--cached','--binary').encode()).hexdigest(),
        original_unstaged_sha256=hashlib.sha256(git('diff','--binary').encode()).hexdigest(),
        original_status=git('status','--porcelain','--untracked-files=no').splitlines(),
        model=MODEL_ID, revision=REVISION, runtime_sha=command(['git','rev-parse','HEAD'], CDB).strip(),
        prior_quality_verdict='QUALITY_FAIL', v3_quality_status='QUALITY_WARNING',
        correction='The v2 primary cap was 4096. A paired union of truncated IDs was extended to 8192. It was not a fresh complete 8192-cap cohort.',
        completed=['Stage 0 physical gate/KV probe', '32 GSM8K + 32 MATH paired quality', 'truncation followup'],
        missing=['diverse Stage 1 pilot','CDB batch cost','live DP4 baselines','validated scheduling oracle'],
        sources={str(p.relative_to(OLD_ROOT)): sha(p) for p in [OLD_ROOT/'reports/final_report.md', OLD_ROOT/'runtime.py',
            OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json']})
    write(target, snapshot)
    write(ROOT/'artifacts/discovery_preregistration.json', dict(
        immutable=True, q=.5, min_depth=2, max_depth=4, kv='last_exited', online=True, replay=False,
        pilot_requests=128, domains={'gsm8k':32,'math500':32,'alpaca':32,'sharegpt':32},
        split='First 8/domain calibration. Remaining 24/domain held-out.', generation_cap=256,
        outputs='Natural EOS permitted. Cap-truncated outputs are not quality evaluations.',
        quality_ids_source='v2 quality_protocol_v3_pilot32.json split=pilot', quality_cap=8192,
        concurrency=[1,2,4,8,16,32,64], arrivals=['burst','closed_loop','poisson'],
        primary_objective='Minimize request P95 E2E. Report throughput and fairness separately.',
        simulator_validation_error_target=.10, predictor_minimum_depth_increment=.10,
        no_quality_stop_for_old_warning=True, no_RL=True, GPUs=[4,5,6,7]))
    write(ROOT/'artifacts/environment_initial.json', dict(
        gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader']),
        topology=command(['nvidia-smi','topo','-m']),
        allowed=[4,5,6,7], DP=4, TP=1, PP=1,
        mapping='One process per physical GPU. Each process sees logical CUDA device 0.'))

if __name__ == '__main__': main()
