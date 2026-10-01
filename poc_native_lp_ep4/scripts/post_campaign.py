"""Bounded extra component coverage and optional producer-readiness control.

The optional cache is a diagnostic lossless encoder-output cache, not a proposed
serving method or a replacement for the uncached primary comparison.
"""
import argparse
import os
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts.finish_campaign import launch, check_correctness


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v4')
    p.add_argument('--producer-control',action='store_true');a=p.parse_args()
    root=Path('poc_native_lp_ep4');v=a.version
    assert read_json(root/'artifacts'/('campaign_status_'+v+'.json'))['status']=='MEASUREMENTS_COMPLETE'
    best=read_json(root/'artifacts/manifests'/('frozen_final_selection_'+v+'.json'))['best_arm']
    # AUTO == G4, so no duplicate component point is required.
    primary={'FULL','CP512','LP_AUTO',best}
    extra=[x for x in ('CP256','CP1024','LP_G8','LP_G16','LP_G24','LP_G48') if x not in primary]
    completed=[]
    if extra:
        completed.append(str(launch('extra_components_'+v,extra,3,trace=True,light=True,decode=128,timeout=2000)))
    if a.producer_control:
        # Cache keys use exact processor tensor bytes/grid; outputs include
        # DeepStack channels. Warmup populates it; no prompt/input/route changes.
        os.environ['NATIVE_LP_VISION_CACHE']='1'
        correct=launch('cached_producer_correctness_'+v,['FULL',best],3,
                       trace=True,light=True,correctness=True,timeout=1500)
        check_correctness(correct,'cached_producer_validation_'+v+'.json')
        completed.append(str(correct))
        completed.append(str(launch('cached_producer_clean_'+v,['CP512',best],3,
                                   paired=True,decode=128,timeout=1800)))
        completed.append(str(launch('cached_producer_components_'+v,['CP512',best],3,
                                   trace=True,light=True,paired=True,decode=128,timeout=1800)))
    atomic_json(root/'artifacts'/('post_campaign_'+v+'.json'),{'status':'COMPLETE',
        'runs':completed,'producer_control':a.producer_control,
        'label':'DIAGNOSTIC_NOT_PRIMARY; no communication method implemented'})


if __name__=='__main__':main()
