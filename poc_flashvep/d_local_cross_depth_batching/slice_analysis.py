"""Summarize actual slice enclosures and native output parity."""
from .common import *
from .analyze import pct,table
from .qualification import measured


def main():
    records=[]
    for model in ('ouro','huginn'):
        folder=ROOT/'raw'/f'slice_costs_{model}_v1'
        if not (folder/'status.json').exists():continue
        reference=measured(ROOT/'raw'/f'local_{model}_native_n8_s2026110701_off_state')[0]
        values=json.loads((folder/'slice_costs.json').read_text())
        checks=[]
        for v in values:
            checks.append(dict(chunk=v['chunk'],quantum=v['quantum'],repetition=v['repetition'],
                output_exact=v['completed']['tokens']==reference['output_ids'][:2],
                fixed_depth_exact=v['completed']['depths']==reference['exit_depths'][:2]))
            if v['warmup']:continue
            groups={}
            for op in v['operations']:
                if not any(s['phase'] in ('RequestPhase.PREFILLING_PRELUDE','RequestPhase.PREFILLING_RECURRENT') for s in op['spans']):continue
                key=(op['operation'],op['loops'])
                groups.setdefault(key,[]).append(op)
            for (operation,loops),ops in groups.items():
                samples=[o['enclosure_ms'] for o in ops]
                row=dict(model=model,chunk=v['chunk'],quantum=v['quantum'],operation=operation,loops=loops,
                    samples=len(samples),warm_repetitions=1,independent_starts=1,
                    full_prompt_recurrent_rows=v['prompt_recurrent_rows'],host_cohort_ms=v['host_cohort_ms'],
                    scope='Serial legal prompt slices. Event enclosures include gaps and graph input costs. No active external decode cohort.')
                for q in (50,95,99):
                    row[f'enclosure_p{q}_ms']=pct(samples,q)
                    row[f'host_submit_p{q}_ms']=pct([o['host_submit_ms'] for o in ops],q)
                row['max_enclosure_ms']=max(samples);records.append(row)
        write(folder/'slice_output_parity.json',dict(passed=all(v['output_exact'] and v['fixed_depth_exact'] for v in checks),
            checks=checks,scope='Serial generated prefix matches the native runtime. N32 state qualification remains a separate gate.'))
    table(ROOT/'slice_costs.csv',records)
    print(json.dumps(dict(rows=len(records))))


if __name__=='__main__':main()
