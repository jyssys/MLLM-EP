"""Do not infer uncapped source retention from a credit4 correctness test."""
import json
import time
from .common import ROOT,write,dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    capsule,sha=freeze('compute_throttle_stock_uncapped400_v1_20261121')
    name=create('Q20_Throttle_StockToken4_C3_UncappedNatural400_v1','Token','longbench_balanced400_v1.json',
        cut=4,pair=(6,7),qps=2.8,seed=2026112201,correctness=True,
        execution_family='INSTALLED_DEFAULT_COMPILE',token_stock_model=True,d_compute_throttle=True,
        continuous_source_credit_limit=0,axis_state_capture=True,axis_capture_hidden_last_only=True,
        gpu_state_digest=True,axis_import_hash_after_model=True,
        source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=4500,serving_timeout=4000)
    print('WAIT_STOCK_CREDIT4_CORRECTNESS',flush=True)
    while True:
        path=ROOT/'state/D_compute_throttle_StockToken_qualification_v1.json'
        if path.exists() and json.loads(path.read_text()).get('passed'):break
        assert dispatch_allowed();time.sleep(5)
    assert execute(name),name
    proof=audit(name);mixed=no_mixed(name);score=quality(name)
    passed=proof['passed'] and proof['decode_full36_observation_complete'] and score['passed']
    passed=passed and all(v['status']=='PASS' for v in proof['imported_prefix_KV'])
    write(ROOT/'state/D_compute_throttle_StockToken_uncapped400_gate_v1.json',dict(
        passed=passed,name=name,n=400,continuous_source_credit_limit=0,
        coverage=proof,compute_deferral=mixed,quality=score,evidence='CORRECTNESS_ONLY',
        scope='Separate full400 stress for the default-compiled stock Token4 path with uncapped source admission. '
            'Actual KV reservation and retirement checks remain enabled. This is not a speed or ITL-safety result.'))
    assert passed
    print('STOCK_THROTTLE_UNCAPPED400_PASS',flush=True)


if __name__=='__main__':main()
