"""Repeat only the incompatible-digest gate with matching observers."""
import json
import torch
from .common import ROOT,write
from .run import save,execute
from .qualification import audit
from .qualification_gate import compare,KV_equal
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    torch.set_num_threads(1)
    old='Q19_Throttle_StockToken4_N8_v1';new='Q19_Throttle_StockToken4_N8_GPUdigest_v2'
    cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text())
    cfg.update(name=new,gpu_state_digest=True,axis_capture_hidden_last_only=True,
        observer_revision='Use the same CPU-verified leaf-SHA256 algorithm as the existing teacher. '
            'The previous raw-SHA256 records cannot be directly compared with leaf-SHA256. '
            'Preserve the old run. Do not classify a digest-schema mismatch as a numerical error.')
    save(cfg)
    path=ROOT/'state/D_compute_throttle_StockToken_qualification_v1.json'
    checkpoint=json.loads(path.read_text());assert len(checkpoint['results'])==1 and not checkpoint['passed']
    results=list(checkpoint['results'])
    write(ROOT/'analysis/stock_throttle_N8_digest_schema_diagnostic_v2.json',dict(
        old=old,replacement=new,old_digest='RAW_SHA256',teacher_digest='SHA256_OF_SHA256_4096B_LEAVES_V1',
        old_hidden_exact_requests=8,old_greedy_exact_requests=8,
        old_full_KV_comparison='INVALID_DIFFERENT_DIGEST_ALGORITHMS',evidence='DIAGNOSTIC',
        scope='Every valid KV byte was observed in both runs,but their digest constructions differ. '
            'Neither equality nor inequality follows from comparing these two roots. '
            'A new run uses the teacher algorithm without changing model computation or correcting tensors.'))
    for index,name in enumerate((new,'Q19_Throttle_StockToken4_C3_Natural32_v1','Q19_Throttle_StockToken4_C3_Natural400_v1')):
        assert execute(name),name
        coverage=audit(name);mix=no_mixed(name)
        numerical=compare('Q13_StockFull36_Teacher4_N8_v1',name) if index==0 else quality(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete'] and numerical['passed']
        assert all(v['status']=='PASS' for v in coverage['imported_prefix_KV'])
        results.append(dict(name=name,coverage=coverage,compute_deferral=mix,numerical=numerical))
        write(path,dict(time=__import__('time').time(),passed=len(results)==4,
            full400_passed=len(results)==4,axis='Token',stock_token_model=True,
            results=results,retained_invalid_digest_run=old,evidence='CORRECTNESS_ONLY'))
    print('STOCK_THROTTLE_MATCHED_DIGEST_QUALIFIED',flush=True)


if __name__=='__main__':main()
