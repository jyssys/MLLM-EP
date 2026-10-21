"""Whole-cohort checkpoint reuse selects by input identity, never by answer or time."""
from .contract import *


def sources():
    rows=dataset(True);selected=[];missing=[]
    for offset in range(0,len(rows),4):
        group=rows[offset:offset+4];index=offset//4;choice=None
        for label in ['quality_S4_pilot_v2','quality_S4_pilot_v1']:
            folder=ROOT/'raw'/label
            if not (folder/f'cohort_{index:03d}_rank0.json').exists():continue
            files=[folder/(r['id'].replace('/','_')+'.json') for r in group]
            if not all(p.exists() for p in files):continue
            values=[read(p) for p in files]
            if all(v['prompt_ids']==r['input_ids'] for v,r in zip(values,group)):
                choice=dict(cohort=index,source_run=label,ids=[r['id'] for r in group],
                    question_files=[str(p.relative_to(ROOT)) for p in files],
                    timing_file=str((folder/f'cohort_{index:03d}_rank0.json').relative_to(ROOT)))
                break
        if choice:selected.append(choice)
        else:missing.append(index)
    return dict(name='quality_S4_canonical',cohorts=selected,missing_cohorts=missing,
        rule='Latest complete comparable cohort, v2 before v1. Choice uses prompt token identity only. No score or latency selection.')


def subset_views():
    """Disjoint pilot and extension sources. No result-based cohort selection."""
    if not (ROOT/'evaluation_subset400.json').exists():return {}
    plan=read(ROOT/'evaluation_subset400.json');rows={r['id']:r for r in dataset(False)}
    pilots={arm:'quality_'+arm+'_pilot_v1' for arm in ARMS}
    pilots['T4']='quality_T4_custom_pilot_v1'
    pilots.update(native_TP1='native_TP1_pilot_v2',native_TP4='native_TP4_pilot_v2')
    single={c['cohort']:c for c in sources()['cohorts']};result={}
    for arm,pilot in pilots.items():
        selected=[];missing=[]
        for cohort in plan['cohorts']:
            index=cohort['index']
            label=pilot if cohort['source']=='pilot' else (
                arm+'_extension400_v1' if arm.startswith('native_') else 'extension400_'+arm+'_v1')
            if arm=='S4' and cohort['source']=='pilot':
                if index in single:selected.append(single[index])
                else:missing.append(index)
                continue
            folder=ROOT/'raw'/label
            timing=folder/f'cohort_{index:03d}_rank0.json'
            if not timing.exists():timing=folder/f'cohort_{index:03d}.json'
            files=[folder/(rid.replace('/','_')+'.json') for rid in cohort['ids']]
            if not timing.exists() or not all(f.exists() for f in files):missing.append(index);continue
            values=[read(f) for f in files]
            assert all(v['prompt_ids']==rows[rid]['input_ids'] for v,rid in zip(values,cohort['ids']))
            # Earlier native pilot rows keep the cap in setup.json, not each answer.
            setup=read(folder/'setup.json') if (folder/'setup.json').exists() else {}
            assert all(v.get('max_new_tokens',setup.get('cap'))==4096 for v in values),(label,index)
            selected.append(dict(cohort=index,source_run=label,ids=cohort['ids'],
                question_files=[str(f.relative_to(ROOT)) for f in files],timing_file=str(timing.relative_to(ROOT))))
        result[arm]=dict(name='quality400_'+arm,cohorts=selected,missing_cohorts=missing,
            rule='Every preselected pilot and extension cohort. Sources are fixed by run label and canonical input identity. No answer or timing selection.')
    return result


if __name__=='__main__':
    value=sources();write(ROOT/'effective_source_manifest.json',value);print('EFFECTIVE_COHORTS',len(value['cohorts']),flush=True)
