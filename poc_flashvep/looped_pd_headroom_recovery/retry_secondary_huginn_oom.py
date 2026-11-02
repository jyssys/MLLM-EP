"""Retain the OOM and test a smaller pool that still fits four long requests."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run
from .primary_campaign import subset_parity
from .async_campaign import integrity


def main():
    source='warm_secondary_aggregated_huginn_l3072_c2048_gate8'
    base=json.loads((ROOT/'configs'/(source+'.json')).read_text())
    base.update(name=source+'_pool25088_retry_v2',kv_blocks=25088,
        OOM_recovery_original=source,
        KV_tuning_scope='Original 25,600-block configuration OOMed on a 104-MiB allocation. A 25,088-block pool saves 1.2890625 GiB and still fits 24,832 reserved blocks plus native watermark. No overcommit or model change.')
    name=put(base,8)
    okay=run(name) and subset_parity('warm_secondary_huginn_l3072_output_gate8',name) and integrity(name)
    if okay:
        configure()
        from ..looped_pd_atomic_prefill import evidence
        evidence.ROOT=ROOT
        check=dict(work=evidence.full_work(name));check['passed']=check['work']['passed']
        write(ROOT/'raw'/name/'recovery_gate.json',check);okay=check['passed']
    ledger=[dict(name=name,passed=okay)]
    if okay:
        cfg=copy.deepcopy(base)
        cfg.update(name='warm_secondary_aggregated_huginn_l3072_c2048_q1p2_pilot24_pool25088_retry_v2',phase='aggregated_QPS_pilot')
        name=put(cfg,24,1.2)
        ledger.append(dict(name=name,passed=run(name) and subset_parity(base['name'],name) and integrity(name)))
    write(ROOT/'state/secondary_huginn_OOM_recovery.json',dict(time=time.time(),stage='COMPLETE_REVIEW_REQUIRED',results=ledger,
        scope='One bounded supported-pool retry only. The original OOM and cleanup error remain preserved. No inference success or speedup is assumed.'))
    restore_if_idle()


if __name__=='__main__':main()
