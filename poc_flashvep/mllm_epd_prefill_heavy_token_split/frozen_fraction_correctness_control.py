"""Test the discovery-frozen fractions before any heldout performance run."""
import asyncio
import json
from pathlib import Path
import random
import time

from . import token_correctness_control as single
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import stop_proxy
from .natural_token_correctness_control import INDICES, cohort
from .source_snapshot import snapshot
from .token_split import select_prefix

TAG='token_FROZEN_C36'


def main():
    preserve();assert time.time()<experiment_deadline()
    discovery=ROOT/'analysis/token_DISCOVERY_C34_summary.json'
    summary=json.loads(discovery.read_text());assert summary['status']=='COMPLETE'
    choice=summary['selection']
    fractions=sorted({choice['fraction'],choice['best_goodput_fraction']})
    for role,gpu in [('E',4),('P',5),('D',6)]:
        lease=json.loads((ROOT/'state'/f'server_token_DISCOVERY_C34_Token_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    rows=[pool[index] for index in INDICES];random.Random(2026112404).shuffle(rows)
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    archive=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_FRACTION_CONFIRMATION',
        source_archive_SHA256=archive['archive_SHA256'],discovery_SHA256=digest(discovery),fractions=fractions,
        requests=[r['request_id'] for r in rows],GPU_group=[4,5,6],evidence='CORRECTNESS_ONLY',
        teacher='Original E+P complete prompt, with the same per-request query boundary as each token fraction.',
        sampling='Natural EOS cap2048,greedy,original BF16 BI0,common compiled policy',
        gates='Strict N1 all28 KV and full greedy output. N8 prefix/suffix numerical gates and full query MRoPE.',
        safety='Suffix starts only after all previous D HTTP streams finish. D protocol errors fail closed.',
        no_heldout_used=True,no_timing_claim=True,full_quality='PENDING'))
    single.TAG=TAG;single.EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    names=[];teacher=None;proxy=None;results=[]
    try:
        single.launch_models(names,fixed_reduction=True,deterministic_reductions=True)
        teacher,teacher_url=single.launch_proxy(TAG+'_teacher_proxy','e_pd',18848,names)
        for fraction in fractions:
            references={};label=f'F{round(100*fraction):02d}'
            for index in INDICES:
                assert time.time()<experiment_deadline()
                row=pool[index];run=f'{TAG}_{label}_N8_{index}_TEACHER'
                prefix=select_prefix(row['total_input_tokens'],fraction)
                single.capture(row,run,teacher_url,[(18841,'P')],prefix)
                references[row['request_id']]=run
            proxy_name=TAG+'_'+label+'_proxy'
            proxy,url=single.launch_proxy(proxy_name,'epd',18849,names,
                suffix_fraction=fraction,token_admission='decode_idle')
            try:
                n1=asyncio.run(cohort(TAG+'_'+label+'_N1',[pool[0]],url,fraction,references,proxy_name))
                n8=asyncio.run(cohort(TAG+'_'+label+'_N8',rows,url,fraction,references,proxy_name))
                results.append(dict(fraction=fraction,N1=n1,N8=n8))
            finally:
                stop_proxy(proxy);proxy=None
        write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),passed=all(
            r['N1']['passed'] and r['N8']['passed'] for r in results),fractions=results,
            evidence='CORRECTNESS_ONLY',task_quality='PENDING',heldout_touched=False))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_fractions=len(results)))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        if teacher is not None:stop_proxy(teacher)
        if names:
            retire([name for name in names if '_E4' not in name]);retire([name for name in names if '_E4' in name])
    preserve()


if __name__=='__main__':main()
