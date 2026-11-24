"""Quiescent one-factor RPC/file/RPC completion-ACK control on Native EPD."""
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

TAG='ec_ACK_C9'
URL='http://127.0.0.1:18847'


def policy(mode,label):
    with urllib.request.urlopen(URL+'/health',timeout=5) as response:state=json.load(response)
    assert all(state[field]==0 for field in ['encoder_pending','prefill_pending','decode_pending'])
    ownership_empty(TAG+'_'+label+'_before_policy','cuda_ipc')
    values={role:rpc(port,'ec_ipc/ack_policy',dict(mode=mode)) for port,role in [(18841,'P'),(18842,'D')]}
    assert all(item['actual']==mode for items in values.values() for item in items)
    write(ROOT/'state'/f'{TAG}_{label}_policy.json',dict(time=time.time(),mode=mode,workers=values,
        scope='Changed only when every coordinator stage was idle and all E-owned exports were released.'))


async def measure(label,mode):
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    run=f'{TAG}_{label}_{mode}_replay100_q2_v0'
    values=await execute(rows,[URL],run,2,2026112402,counts,metadata=dict(evidence='DIAGNOSTIC_C3',
        manifest_SHA256=digest(path),EC_transport='cuda_ipc',completion_ACK=mode,
        scope='Same initialization and original Native EPD. Only the completion-ACK path changes. Not token-split serving.'))
    assert len(values)==100 and all(row['status']=='MEASURED_ONLINE' for row in values)
    ownership_empty(run,'cuda_ipc')
    value=summarize(run)
    print('ACK_CONTROL_ARM',run,{key:value[key] for key in ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)
    return value


def main():
    preserve();archive=snapshot('native_IPC_ACK_control_v0')
    plan=ROOT/'configs/native_IPC_ACK_control_v0.json';assert not plan.exists()
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_ACK_INTERVENTION',source_archive_SHA256=archive['archive_SHA256'],
        evidence='DIAGNOSTIC_CONTROL_PLAN',one_factor='P waits for an E completion-ACK RPC versus writes an owned completion marker',
        baseline='ec_IPC_C8 preserves correct N1/N8/all28 KV, but equal-output q2 TTFTp95 regresses7.15s->11.54s.',
        hypotheses=['An unnecessary synchronous ACK can delay P even after its GPU clone is complete.',
            'If export RPC dominates, changing only the completion ACK will not recover the CPU baseline.',
            'Device-open and clone costs can remain after both control-plane delays are reduced.'],
        arms=['RPC_A','FILE_B','RPC_C'],independent_starts=1,not_N3=True,
        fixed=['Same GPU4/5/6 group and initialization','Main100','Equal5269 output tokens','QPS2',
            'Arrival seed2026112402/order seed2026112404','All model,processor,B/S,KV,compiler and SLO settings'],
        mandatory='CUDA clone completion synchronization and producer ownership release gates remain in both modes. '
            'A file marker cannot free a live tensor before the downstream request finishes.',
        interpretation='Conservative transport control. No offloaded prefill, future window predictor or dynamic token selection.'))
    names=[];proxy=None
    try:
        names,proxy=launch(TAG,'cuda_ipc')
        asyncio.run(correctness(TAG+'_RPC','cuda_ipc'))
        policy('file','FILE_GATE')
        asyncio.run(correctness(TAG+'_FILE','cuda_ipc'))
        kv_capture(TAG+'_FILE','cuda_ipc');compare_kv('ec_CPU_C6',TAG+'_FILE')
        policy('rpc','WARMUP')
        asyncio.run(measure('WARMUP','rpc'))
        results=[]
        for label,mode in [('A','rpc'),('B','file'),('C','rpc')]:
            policy(mode,label);results.append(asyncio.run(measure(label,mode)))
        write(ROOT/'analysis/native_IPC_ACK_control_v0.json',dict(time=time.time(),rows=results,
            evidence='DIAGNOSTIC_C3',independent_starts=1,confidence_interval='NOT_AVAILABLE',quality='PENDING',
            scope='RPC/file/RPC on one Native EPD initialization. A lower TTFT is not proof of token-split gain.'))
    finally:
        if proxy is not None:stop_proxy(proxy)
        if names:retire(names)
    preserve()


if __name__=='__main__':main()
