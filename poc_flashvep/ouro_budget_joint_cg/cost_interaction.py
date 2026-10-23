"""Exact endpoint interaction, with process/GPU blocks as replication units."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file
from .metrics import paired_block_interval

def main():
    rows=[];full=[]
    for folder in sorted((ROOT/'raw').glob('cb_p2_controlled_*')):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('*_component_T*.json'):
            data=json.loads(file.read_text());P=data['P'];C=data['C']
            core=[r for r in data['service_observations'] if any(s['phase']=='prefilling_recurrent' for s in r['spans'])]
            opening=[r for r in data['service_observations'] if any(s['phase']=='prefilling_prelude' and s['start']+s['tokens']==P for s in r['spans'])]
            endpoint=[r for r in core if r['index'] in data['endpoint_rows']]
            for q in endpoint:
                span=q['spans'][0]
                assert len(q['spans'])==1 and span['start']+C==P and span['tokens']==C
                rows.append(dict(run_id=data['run_id'],GPU=data['physical_GPU'],P=P,C=C,G=span['G'],
                    depth=span['depth'],trial=data['trial'],service_ms=q['service_ms'],
                    path='staged_core',context=P,policy=data['policy'],
                    evidence='MEASURED_OBSERVER_COMPONENT_EXACT_ENDPOINT_SHARED_HOST'))
            intervals=sorted((r['aligned_service_begin_s'],r['aligned_service_end_s']) for r in endpoint+opening)
            full.append(dict(run_id=data['run_id'],GPU=data['physical_GPU'],P=P,C=C,policy=data['policy'],trial=data['trial'],
                sum_endpoint_stream_service_ms=sum(r['service_ms'] for r in endpoint+opening),
                endpoint_dispatches=len(endpoint)+len(opening),completed_prompt_tokens=C,
                gap_between_first_and_last_endpoint_ms=1000*(intervals[-1][1]-intervals[0][0]),
                warning='Service sum is a complete-chunk component cost, not a measured E2E saving.'))
    csv_file(ROOT/'controlled_endpoint_service.csv',rows);csv_file(ROOT/'full_chunk_completion_cost.csv',full)
    block=defaultdict(list)
    for r in rows:
        if r['depth']==0:
            block[(r['run_id'],r['P'],r['C'],r['G'])].append(r['service_ms'])
    med={key:float(np.median(values)) for key,values in block.items()}
    effects=[]
    for P in (2048,8192):
        for c1,c2 in ((128,512),(512,2048),(128,2048)):
            for g1,g2 in ((1,2),(2,4),(1,4),(2,3)):
                differences=[]
                for run in sorted({r['run_id'] for r in rows if r['P']==P}):
                    keys=[(run,P,c,g) for c,g in ((c2,g2),(c1,g2),(c2,g1),(c1,g1))]
                    if all(key in med for key in keys):
                        a,b,c,d=[med[key] for key in keys]
                        differences.append((0.,a-b-c+d))
                interval=paired_block_interval(differences)
                if interval:
                    effects.append(dict(P=P,C1=c1,C2=c2,G1=g1,G2=g2,**interval,
                        interpretation='Conditional nonseparability at exact context endpoint and depth zero.',
                        evidence='MEASURED_COMPONENT_INTERACTION_NOT_ONLINE_GAIN'))
    csv_file(ROOT/'interaction_effects.csv',effects)
    model_errors=[]
    def features(C,G,mode):
        if mode=='ADDITIVE_CATEGORICAL':
            return [1.,float(C==512),float(C==2048),float(G==2),float(G==3),float(G==4)]
        if mode=='PRODUCT_ONLY':
            return [1.,C*G/2048.]
        return [float(C==c and G==g) for c in (128,512,2048) for g in (1,2,3,4)]
    for P in (2048,8192):
        for mode in ('ADDITIVE_CATEGORICAL','PRODUCT_ONLY','INTERACTIVE_CATEGORICAL'):
            blocks=sorted({key[0] for key in med if key[1]==P})
            for test_block in blocks:
                training=[(key,value) for key,value in med.items() if key[1]==P and key[0]!=test_block]
                test=[(key,value) for key,value in med.items() if key[1]==P and key[0]==test_block]
                if not training or not test:
                    continue
                x=np.array([features(k[2],k[3],mode) for k,v in training]);y=np.array([v for k,v in training])
                coefficients=np.linalg.lstsq(x,y,rcond=None)[0]
                errors=[v-float(np.dot(features(k[2],k[3],mode),coefficients)) for k,v in test]
                model_errors.append(dict(P=P,model=mode,test_process=test_block,n_cells=len(errors),
                    MAE_ms=float(np.mean(np.abs(errors))),P95_abs_error_ms=float(np.percentile(np.abs(errors),95)),
                    scope='Leave-one-GPU/process-out component cell medians. Same three input families, not blind serving prediction.'))
    csv_file(ROOT/'cost_separability_models.csv',model_errors)
    print('Endpoint records',len(rows),'complete-chunk component rows',len(full),'paired interactions',len(effects))

if __name__=='__main__':
    main()
