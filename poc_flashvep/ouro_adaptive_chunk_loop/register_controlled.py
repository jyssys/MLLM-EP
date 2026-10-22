"""Register matched-prefix controls before examining new diagnostic tensors."""
import json
from .common import ROOT,write

def main():
    audit=json.loads((ROOT/'legacy_parity_registration.json').read_text())
    reference=next(r for r in audit['mismatches'] if r['id']=='bg_11_3')
    jobs=[]
    for gpu,graphs in [(4,True),(5,False)]:
        jobs.append(dict(id=f'p1_common_prefix_graph{int(graphs)}_v1',kind='diagnostic',gpu=gpu,
            target='bg_11_3',trial=11,stop_index=8,kv_steps=[0,8],modes=['atomic','grouped','split'],layers=list(range(24)),
            budget=1024,blocks=12288,graphs=graphs,streams=True,split='DIAGNOSTIC',timeout_s=3600,
            prefix_reference_path=reference['reference'],prefix_reference_sha256=reference['reference_sha256']))
    jobs.append(dict(id='p1_known_legacy_P2048_N4_T10_v1',kind='continuity',gpu=6,
        shape=[2048,4],cells=[['atomic',1024],['split',1024],['grouped',1024],['mixed',512]],trials=[10],budget=1024,
        split='LEGACY',timeout_s=1800))
    path=ROOT/'p1_controlled_plan.json'
    assert not path.exists(),'Controlled diagnostic registration is immutable'
    write(path,dict(gpus=list(range(8)),jobs=jobs,
        scope='Common frozen input/output prefix numerical controls only. Never a quality or performance result.',
        untouched_original_plan='p1_diagnostic_plan.json'))

if __name__=='__main__':
    main()
