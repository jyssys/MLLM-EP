"""Refer to previous committed live-only archives without recapturing/copying them."""
import json
from poc_dllm_request_persistent_replica.contract import *


def main():
    old=json.loads((SHORT/'results/reused_trace_manifest.json').read_text())
    split=json.loads((SHORT/'results/design_freeze.json').read_text())
    rows=[]
    for row in old['requests']:
        if row['block_length']!=32:continue
        row=dict(row);row['archive']=str(SHORT/'data/live_routes'/(row['request']+'.npz'))
        row['split']='calibration' if row['request_id'] in split['calibration'] else 'heldout'
        row['lane']='short';row['archive_sha256']=sha(Path(row['archive']))
        rows.append(row)
    new_json(ROOT/'results/short_manifest.json',dict(requests=rows,parent=PARENT,
        continuity='existing64-budget short traces;2/3blocks insufficient primary persistent lifetime but valid control',
        changed_source_artifacts=False,new_gpu_requests=0))


if __name__=='__main__':main()
