"""Additive E2E percentages from whole-rollout critical-stream intervals."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write

ALIASES={
    'input_staging':'Input staging',
    'metadata_pack':'EP metadata and count synchronization',
    'count_allgather':'EP metadata and count synchronization',
    'metadata_and_count_sync':'EP metadata and count synchronization',
    'dispatch':'Dispatch including rank wait',
    'receive_layout':'Expert and local tensor layout',
    'expert_compute':'Expert and local tensor layout',
    'return_layout':'Expert and local tensor layout',
    'expert_and_local_layout':'Expert and local tensor layout',
    'expert_native_including_layout':'Expert and local tensor layout',
    'combine':'Combine including rank wait',
    'weighting_layout':'Reweight and restore including rank wait',
    'restore_allreduce':'Reweight and restore including rank wait',
    'reweight_and_restore':'Reweight and restore including rank wait',
    'attention_qkv_rope':'Attention including QKV and RoPE',
    'router':'Router',
    'moe_block_other':'Block normalization residual and auxiliary work',
    'lm_head':'LM head',
    'embedding':'Embedding',
    'normalization':'Other model and scheduling',
    'model_other':'Other model and scheduling',
    'other_ep':'Other model and scheduling',
    'input_update':'Other model and scheduling',
    'sampling_output':'Sampling output and completion synchronization',
    'unattributed':'Unattributed',
}
COMPONENTS=tuple(dict.fromkeys(ALIASES.values()))

def main():
    raw=[];groups=defaultdict(list);phase_rows=[];rng=np.random.default_rng(20261026)
    for directory in sorted((ROOT/'raw').glob('*_e2e_share_*_v1')):
        config_path=ROOT/'runs/configs'/f'{directory.name}.json'
        if not config_path.exists():continue
        config=json.loads(config_path.read_text());ep=len(config['gpus'])
        status=directory/'attribution_status.json'
        if not status.exists():continue
        cells={(v['batch'],v['prefix'],v['policy']):v for v in json.loads(status.read_text())}
        for path in directory.glob('attribution_*.events.json.gz'):
            marker=path.with_name(path.name.replace('.events.json.gz','.markers.json'))
            if not marker.exists():continue
            meta=json.loads(marker.read_text())
            with gzip.open(path,'rt') as source:data=json.load(source)
            key=(meta['batch'],meta['prefix'],meta['policy'])
            if key not in cells:continue
            cell=cells[key];window=data['window_ms'];primary=defaultdict(float)
            for label,ms in data['exclusive_components_ms'].items():
                primary[ALIASES.get(label,'Unattributed')]+=ms
                raw.append(dict(model=config['model'],ep=ep,batch=key[0],prefix_tokens=key[1],policy=key[2],
                    repeat=meta['repeat'],component=label,component_ms=ms,component_pct=100*ms/window,
                    observed_e2e_gpu_span_ms=window,observed_host_e2e_ms=data['host_e2e_ms'],
                    clean_host_e2e_mean_ms=float(np.mean(cell['clean_e2e_ms'])),overhead_ratio=cell['overhead_ratio'],
                    low_overhead_timing_gate=cell['low_overhead_timing_gate'],detail=cell['detail'],allocator_conf=config.get('allocator_conf','DEFAULT'),
                    source=path.relative_to(ROOT).as_posix(),evidence='LIVE_WHOLE_E2E_EXCLUSIVE_CRITICAL_STREAM_SPANS'))
            assert abs(sum(primary.values())-window)<1e-5
            for component in COMPONENTS:primary.setdefault(component,0.)
            for component,ms in primary.items():
                groups[(config['model'],ep,*key,cell['detail'],cell['low_overhead_timing_gate'],config.get('allocator_conf','DEFAULT'),component)].append(dict(ms=ms,pct=100*ms/window,window=window,
                    host=data['host_e2e_ms'],clean=float(np.mean(cell['clean_e2e_ms'])),
                    overhead=cell['overhead_ratio'],gate=cell['low_overhead_timing_gate'],source=directory.name))
            for phase,values in data.get('phase_components',{}).items():
                components=defaultdict(float)
                for label,ms in values['components_ms'].items():components[ALIASES.get(label,'Unattributed')]+=ms
                assert abs(sum(components.values())-values['window_ms'])<1e-5
                for component in COMPONENTS:
                    ms=components.get(component,0.)
                    phase_rows.append(dict(model=config['model'],ep=ep,batch=key[0],prefix_tokens=key[1],policy=key[2],
                        detail=cell['detail'],allocator_conf=config.get('allocator_conf','DEFAULT'),repeat=meta['repeat'],phase=phase,component=component,component_ms=ms,
                        component_pct_of_whole_e2e=100*ms/window,component_pct_of_phase=100*ms/values['window_ms'],
                        phase_ms=values['window_ms'],phase_pct_of_whole_e2e=100*values['window_ms']/window,
                        low_overhead_timing_gate=cell['low_overhead_timing_gate'],source=path.relative_to(ROOT).as_posix(),
                        evidence='LIVE_EXCLUSIVE_ENDPOINT_STREAM_PHASE_ATTRIBUTION'))
    output=[]
    for key,values in groups.items():
        model,ep,b,prefix,policy,detail,gate,allocator,component=key
        percentages=np.asarray([v['pct'] for v in values])
        low=high='NOT_AVAILABLE_SINGLE_TRACE'
        if len(values)>1:
            bootstrap=rng.choice(percentages,(4000,len(values)),replace=True).mean(1)
            low=float(np.quantile(bootstrap,.025));high=float(np.quantile(bootstrap,.975))
        output.append(dict(model=model,ep=ep,batch=b,prefix_tokens=prefix,policy=policy,detail=detail,allocator_conf=allocator,component=component,
            mean_component_ms=float(np.mean([v['ms'] for v in values])),mean_component_pct=float(percentages.mean()),
            ci95_low_pct=low,ci95_high_pct=high,traced_rollouts=len(values),
            observed_e2e_gpu_span_mean_ms=float(np.mean([v['window'] for v in values])),
            observed_host_e2e_mean_ms=float(np.mean([v['host'] for v in values])),
            clean_host_e2e_mean_ms=float(np.mean([v['clean'] for v in values])),
            observer_overhead_mean_ratio=float(np.mean([v['overhead'] for v in values])),
            timing_status='LOW_OVERHEAD_OBSERVER' if gate else 'OBSERVER_ONLY_TIMING_GATE_FAILED',
            run_ids='|'.join(sorted({v['source'] for v in values})),
            ci_scope='Rollouts within the same process start. No independent-start confidence claim.',
            evidence='LIVE_EXCLUSIVE_ENDPOINT_STREAM_ATTRIBUTION_NOT_KERNEL_SERVICE_SUM',
            qualification='API execution spans include launch gaps and waits. Shares describe instrumented E2E. Zero means no separately attributed interval, not no arithmetic.'))
    csv_write(ROOT/'e2e_component_percentages_raw.csv',raw)
    csv_write(ROOT/'e2e_component_percentages.csv',output)
    csv_write(ROOT/'e2e_component_phase_percentages.csv',phase_rows)
    from .attribution_report import render
    render(output)

if __name__=='__main__':main()
