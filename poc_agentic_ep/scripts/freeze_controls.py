"""Controls chosen from metadata before held-out routing is inspected."""
import json
import math
from poc_agentic_ep.contract import ROOT,write,sha


def main():
    path=ROOT/'results/frozen_controls.json'
    if path.exists():raise FileExistsError(path)
    manifest=ROOT/'results/frozen_manifest_v2.json';data=json.loads(manifest.read_text())
    rows=data['rows'];pairs=[]
    by={(r['trajectory_id'],r['turn']):r for r in rows}
    for target in rows:
        if target['split']!='heldout' or target['turn']==0:continue
        prev=by[target['trajectory_id'],target['turn']-1]
        same=[r['request_id'] for r in rows if r['trajectory_id']==target['trajectory_id'] and abs(r['turn']-target['turn'])>1]
        other=[r for r in rows if r['trajectory_id']!=target['trajectory_id']]
        other.sort(key=lambda r:(abs(math.log(r['new_suffix_tokens']/prev['new_suffix_tokens'])),r['request_id']))
        matched=[];used=set()
        for r in other:
            if r['trajectory_id'] in used:continue
            used.add(r['trajectory_id']);matched.append(dict(request_id=r['request_id'],
                suffix_length_ratio=r['new_suffix_tokens']/prev['new_suffix_tokens']))
            if len(matched)==3:break
        pairs.append(dict(trajectory_id=target['trajectory_id'],previous=prev['request_id'],target=target['request_id'],
            nonadjacent_sources=same,cross_trajectory_sources=matched))
    assert len(pairs)==64
    write(path,dict(manifest_sha256=sha(manifest),pairs=pairs,
        same_control='mean over all nonadjacent source turns; diagnostic may include future turns, never a deployable predictor',
        cross_control='three closest metadata-length sources in different trajectories; chosen without any route, score or generated-output information',
        global_baseline='only four disjoint calibration trajectories, fully completed before held-out',
        selection='no current next-turn truth used by previous/global policies'))
    print('FROZEN_64_CONTROL_PAIRS')


if __name__=='__main__':main()
