"""Freeze one causal control after new profiling and architecture screening."""
import datetime
from poc_dllm_multigpu_e2e_atlas.contract import *


def main():
    target=ROOT/'manifests/frozen_control.json';assert not target.exists()
    audit=json.loads((ROOT/'manifests/profile_validity.json').read_text())
    assert all(r['valid_bounded'] for r in audit['torch_profile'])
    decision=ROOT/'results/architecture_decision.json';assert decision.exists()
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_prefix=16384,secondary_calibration_prefix=4096,
        calibration_IDs=[r['request_id'] for r in rows()],heldout_IDs=[r['request_id'] for r in rows('held_out')],
        selected='Compact out-of-place KV scatter. No in-place mutation of the native source cache.',
        one_factor='Clone only the logical layer KV tensor, then write fresh canvas to that independent clone.',
        unchanged=['TP4/EP1/DP1/PP1/CP1','BF16','weights','router','attention mask','RoPE','sampler','threshold','EOS','block32','output64'],
        source_proof='Actual source NVTX + CUPTI D2D bytes confirm8MiB logical view triggers320MiB backing-storage clone.',
        physical_gate='Low-overhead all-occurrence trace shows about123ms exclusive copy activity in about432ms profile. Fine DAG still PARTIAL. This trial tests actual removability.',
        architecture_screen_sha256=sha(decision),profile_validity_sha256=sha(ROOT/'manifests/profile_validity.json'),
        repeats=dict(warmup=3,clean=3,coarse=1,parity_only=1),
        discrete_parity='EXACT output IDs, acceptance updates, NFE, EOS/stop interpretation',
        tensor_parity='EXACT source unchanged, exact scatter values. Check final graph KV and logits on deterministic probes.',
        bootstrap='2000 paired request resamples. Technical repeats are not independent requests.',
        novelty='Generic exact storage/layout correction. Do not claim new parallel architecture from speedup alone.',
        no_further_candidates=True))
    print('Frozen one control and untouched24 held-out IDs before control results.')


if __name__=='__main__':main()
