"""Freeze the diagnostic rotations before collecting new numerical results."""
import json
import random
from .common import ROOT, write

def main():
    jobs=[]
    trials=[2,3,5,6,8,9,10,11]
    for gpu,trial in enumerate(trials):
        jobs.append(dict(id=f'p1_legacy_reproduction_t{trial}_g{gpu}',kind='continuity',gpu=gpu,
            shape=[8192,8],cells=[['atomic',1024],['grouped',1024],['split',1024],['mixed',512]],
            trials=[trial],budget=1024,graphs=True,split='LEGACY',timeout_s=1800))
    controls=[(True,True),(False,True),(False,False),(True,False)]
    for i,(graphs,streams) in enumerate(controls):
        jobs.append(dict(id=f'p1_first_divergence_graph{int(graphs)}_stream{int(streams)}_v1',kind='diagnostic',gpu=i,
            target='bg_11_3',trial=11,stop_index=8,kv_steps=[0,8],modes=['atomic','grouped','split'],
            layers=list(range(24)),budget=1024,blocks=12288,graphs=graphs,streams=streams,
            split='DIAGNOSTIC',timeout_s=3600))
    for i,chunk in enumerate([64,128,256,512,768,1024,1536,2048]):
        jobs.append(dict(id=f'p1_isolated_C{chunk}_v1',kind='qualify',gpu=(i+4)%8,chunk=chunk,
            lengths=[128,512,2048,8192],budget=2048,split='DIAGNOSTIC',timeout_s=5400))
    write(ROOT/'p1_diagnostic_plan.json',dict(gpus=list(range(8)),jobs=jobs,
        prerequisite='Review Phase0 identity/ownership and interference before dispatch. Heavy diagnostics are not clean serving measurements.'))
    # Ordered discovery, not test-outcome-guided cell selection.
    shapes=[(512,1),(2033,4),(8177,8)]
    modes=['atomic','split','grouped','grouped3','staged4','mixed']
    chunks=[64,128,256,512,768,1024,1536,2048]
    matrix=[]
    for shape_index,shape in enumerate(shapes):
        for chunk_index,chunk in enumerate(chunks):
            for split_index,split in enumerate(['TRAIN','VALIDATION','CALIBRATION','HELDOUT']):
                order=list(modes)
                random.Random(1023+shape_index*100+chunk_index*10+split_index).shuffle(order)
                gpu=(shape_index*3+chunk_index+split_index*2)%8
                matrix.append(dict(id=f'p2_P{shape[0]}_N{shape[1]}_C{chunk}_{split.lower()}_v1',
                    kind='surface',gpu=gpu,shape=shape,cells=[[m,chunk] for m in order],
                    split=split,trials=[0,1,2],cap=32,budget=2048,trace=True,timeout_s=5400,
                    prereq='G3 isolated qualification. Real state bugs block affected paths.'))
    write(ROOT/'p2_surface_plan.json',dict(gpus=list(range(8)),jobs=matrix,
        status='REGISTERED_NOT_AUTHORIZED_UNTIL_G0_G1_REVIEW',
        selection='Complete three-shape C64..2048 policy grid. Separate source-family and process groups.'))
    print('REGISTERED_P1',len(jobs),'P2',len(matrix))

if __name__=='__main__':
    main()
