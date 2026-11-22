"""Test output interventions on an exact iso-work continuous-serving frontier."""
import argparse
import json
import time
from .common import ROOT,write,digest
from .c3_campaign import qualified_registry,performance,launch,wait_file

def freeze_sets():
    from ..dense_ar_longbench400_pd.mechanism_sets import exact_subset
    from ..dense_ar_longbench400_pd.geometry import matched_suffixes
    path=ROOT/'data/c3_iso8_original_exact16_sets_v2.json';assert not path.exists();sets=[]
    for bucket in (0,3):
        source=ROOT/'data'/f'mechanism_input{bucket}_heldout.json';rows=json.loads(source.read_text())
        chosen,feasible=exact_subset(rows,modulus=1152);assert feasible
        total=sum(r['input_tokens'] for r in chosen);token,work=matched_suffixes([r['input_tokens'] for r in chosen],8)
        assert work['exact_match']
        hybrid=[r['input_tokens']//128*16 for r in chosen];remainder=total//8-sum(hybrid);assert remainder%16==0
        order=sorted(range(len(chosen)),key=lambda i:(-(chosen[i]['input_tokens']/8-hybrid[i]),i))
        for i in order[:remainder//16]:hybrid[i]+=16
        assert 4*total+32*sum(hybrid)==8*total
        dataset=f'c3_iso8_input{bucket}_heldout_exact16_v2.json';write(ROOT/'data'/dataset,chosen)
        sets.append(dict(input_bucket=bucket,dataset=dataset,SHA256=digest(ROOT/'data'/dataset),input_sum=total,n=16,
            Layer_cut=8,Token_cut=8,Hybrid_cut=4,Hybrid_fraction=.125,P_rows=28*total,D_rows=8*total,
            Hybrid_suffix_by_request={f'q{i}':v for i,v in enumerate(hybrid)},Token_suffix_by_request={f'q{i}':v for i,v in enumerate(token)}))
    write(path,dict(time=time.time(),sets=sets,evidence='STRUCTURAL_ACCOUNTING',
        scope='Select unchanged heldout originals by deterministic identity and input-length residue only. '
            'No performance result selects these sets. All three axes have exact aggregate P/D GEMM rows. '
            'Attention distribution,hidden payload and continuous batch shapes remain different.'))
    print('C3_EXACT_ISO_FRONTIER_ORIGINAL_REQUESTS_FROZEN',flush=True)

def prepare():
    sets=json.loads((ROOT/'data/c3_iso8_original_exact16_sets_v2.json').read_text())['sets'];registry=qualified_registry()
    required=('NativeStrong','NativeCommon2','Layer8','StockToken8','Hybrid4_f0125')
    assert all(k in registry for k in required),('C3 bridge requires fresh qualified arms',set(required)-set(registry))
    actions={k:registry[v] for k,v in dict(NativeStrong='NativeStrong',NativeCommon2='NativeCommon2',Layer='Layer8',StockToken='StockToken8',Hybrid='Hybrid4_f0125').items()}
    qps=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text())['reference_knee']
    lanes=[[],[],[],[]];blocks=[]
    for index,row in enumerate(sets):
        lane=2+index;pair=[2*lane,2*lane+1]
        for output in (16,192):
            for start in range(3):
                arms={};seed=2026112291+start
                order=('NativeStrong','Layer','StockToken','Hybrid','NativeCommon2') if start%2 else ('Hybrid','StockToken','Layer','NativeCommon2','NativeStrong')
                for label in order:
                    name=f'V2_C3Iso8_I{row["input_bucket"]}_O{output}_s{start}_{label}'
                    override=row['Hybrid_suffix_by_request'] if label=='Hybrid' else row['Token_suffix_by_request'] if label=='StockToken' else None
                    performance(actions[label],name,row['dataset'],pair,qps,seed,output=output,suffix_override=override)
                    lanes[lane].append(name);arms[label]=name
                blocks.append(dict(split=f'Iso8_I{row["input_bucket"]}_O{output}',qps=qps,start=start,seed=seed,pair=pair,dataset=row['dataset'],
                    input_bucket=row['input_bucket'],output=output,arms=arms,
                    NativeCommon_for={k:'NativeCommon2' for k in arms},exact_work=row))
    path=ROOT/'jobs/c3_continuous_iso8_output_bridge_v2.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,
        scope='Fixed-output control,not natural EOS. Keep Input,cut,exact aggregate work,arrival and admission constant across Output16/192. '
            'Use the original compiled Stock Token baseline,compiled Layer prefix and genuine Hybrid. '
            'All axis state/quality compositions were validated before these timings. '
            'Continuous actual batch shapes are observed,not forced to C0 batch1. Point differences do not identify one queue mediator.'))
    return path.name

def run():
    wait_file(ROOT/'state/c0_campaign_complete_v2.json');wait_file(ROOT/'state/c3_pilot_and_geometry_complete_v2.json')
    name=prepare();launch(name)
    write(ROOT/'state/c3_continuous_iso8_output_bridge_complete_v2.json',dict(time=time.time(),complete=True,plan=name))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze_sets','run'));args=p.parse_args();globals()[args.action]()
