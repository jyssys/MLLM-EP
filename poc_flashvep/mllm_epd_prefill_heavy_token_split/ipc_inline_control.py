"""Native-only early descriptor publication, with the same clone and free gates."""
import asyncio
import json
import random
import time
import urllib.request

from .client import execute,resolve_manifest,load_replay_counts
from .common import ROOT,digest,preserve,write
from .ipc_ec_control import launch,stop_proxy,correctness,kv_capture,compare_kv,ownership_empty
from .frontend_supply_control import retire
from .llm_probe_control import rpc
from .metrics import summarize
from .source_snapshot import snapshot

TAG='ec_INLINE_C10';URL='http://127.0.0.1:18847'


def inline(enabled,label):
    with urllib.request.urlopen(URL+'/health',timeout=5) as response:value=json.load(response)
    assert all(value[k]==0 for k in ['encoder_pending','prefill_pending','decode_pending'])
    ownership_empty(TAG+'_'+label+'_quiescent','cuda_ipc')
    result=rpc(18840,'ec_ipc/inline_policy',dict(enabled=enabled))
    assert len(result)==1 and result[0]['inline_role']==('prefill' if enabled else None)
    write(ROOT/'state'/f'{TAG}_{label}_policy.json',dict(time=time.time(),enabled=enabled,workers=result))


async def measure(label,enabled):
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests'];random.Random(2026112404).shuffle(rows)
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    run=f'{TAG}_{label}_inline{int(enabled)}_replay100_q2_v0'
    values=await execute(rows,[URL],run,2,2026112402,counts,metadata=dict(evidence='DIAGNOSTIC_C3',
        manifest_SHA256=digest(path),EC_transport='cuda_ipc',completion_ACK='file',inline_native_descriptors=enabled,
        scope='Native full-P only. This does not implement a token-prefix continuation.'))
    assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
    ownership_empty(run,'cuda_ipc');value=summarize(run)
    print('INLINE_CONTROL_ARM',run,{k:value[k] for k in ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)
    return value


def main():
    preserve();archive=snapshot('native_IPC_inline_control_v0')
    plan=ROOT/'configs/native_IPC_inline_control_v0.json';assert not plan.exists()
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_INLINE_RESULTS',source_archive_SHA256=archive['archive_SHA256'],
        one_factor='Publish Native P descriptors with the E result, rather than after a later P-to-E export RPC.',
        prior_ACK_result='Same-start RPC/file/RPC TTFTp95 11.81/9.11/12.49s. File ACK remains slower than CPU7.15s.',
        prior_cost='File ACK P-load mean exportRPC78.405ms, import/clone/sync145.992ms, completionACK0.288ms.',
        arms=['ON_DEMAND_A','INLINE_B','ON_DEMAND_C'],GPU_group=[4,5,6],QPS=2,requests=100,output_tokens=5269,
        independent_starts=1,not_N3=True,
        safety='Clone completion and producer lifetime stay unchanged. Each inline descriptor is consumed once. '
            'A cache reload requires a fresh on-demand descriptor. Failed requests cannot trigger unsafe early free.',
        limit='Only Native full-P uses this control. Generic token split requires an explicit per-image consumer coverage plan.',
        fixed='Same model,processor,BI0,weights,KV,B/S,arrival,output contract,file ACK and SLO. No extra prefill insertion.',
        interpretation='Native runtime root-cause control, not a new modality-aware method or token-split serving gain.'))
    names=[];proxy=None
    try:
        names,proxy=launch(TAG,'cuda_ipc')
        for port in [18841,18842]:rpc(port,'ec_ipc/ack_policy',dict(mode='file'))
        inline(False,'OFF_GATE');asyncio.run(correctness(TAG+'_OFF','cuda_ipc'))
        inline(True,'ON_GATE');asyncio.run(correctness(TAG+'_ON','cuda_ipc'))
        kv_capture(TAG+'_ON','cuda_ipc');compare_kv('ec_CPU_C6',TAG+'_ON')
        inline(False,'WARMUP');asyncio.run(measure('WARMUP',False))
        results=[]
        for label,enabled in [('A',False),('B',True),('C',False)]:
            inline(enabled,label);results.append(asyncio.run(measure(label,enabled)))
        write(ROOT/'analysis/native_IPC_inline_control_v0.json',dict(time=time.time(),rows=results,
            evidence='DIAGNOSTIC_C3',independent_starts=1,confidence_interval='NOT_AVAILABLE',quality='PENDING',
            scope='One-factor early publication control. Not a proof of generic exact token-split speedup.'))
    finally:
        if proxy is not None:stop_proxy(proxy)
        if names:
            # Stop all consumer contexts before the owner of their CUDA storage.
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__=='__main__':main()
