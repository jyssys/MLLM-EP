"""Audit the invalid record instead of discarding it from capacity evidence."""
import json
from .common import *


def main():
    initialize();old=json.loads((ROOT/'state/graph_bound_18k_plan.json').read_text());jobs=[]
    for parent in old['all_jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('graph_bound_','graph_sync_')
        cfg.update(name=name,activity_full_device_sync_after_outputs=True,
            phase='POST_OUTPUT_ALL_STREAM_SYNC_INVALID_RECORD_DIAGNOSIS')
        if parent==old['state8']:
            cfg['required_qualification']='graph_bound_ouro_L18432_observer_state8'
        elif cfg.get('discovery_activity'):
            cfg['required_qualification']=old['state8'].replace('graph_bound_','graph_sync_')
        save(cfg);jobs.append(name)
    write(ROOT/'state/graph_sync_18k_plan.json',dict(state8=jobs[0],all_jobs=jobs,
        clean_reference=jobs[1],jobs=jobs[2:],
        scope='All-device synchronization occurs only after Serving outputs. Invalid records remain fatal. Their raw fields now remain in a separate diagnostic CSV. No record is discarded to pass coverage.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
