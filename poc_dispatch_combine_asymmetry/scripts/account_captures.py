import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from poc_dispatch_combine_asymmetry.accounting import account
from poc_dispatch_combine_asymmetry.contract import write


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();records=[]
    for run in args.runs:
        groups=collections.defaultdict(dict)
        for path in (run/'captures').glob('*.pt'):
            x=torch.load(path,map_location='cpu',weights_only=False)
            r=x['record'];groups[(r['wave'],r['layer'])][r['ep_rank']]=x
        for (wave,layer), data in groups.items():
            assert set(data)==set(range(4))
            r=data[0]['record'];n=r['prompt_tokens'];width=(n+1)//2
            logical=[];executed=[];ownership=[]
            for rank in range(4):
                ids=data[rank]['ids'].numpy()
                rows=min(width,max(0,n-rank*width)) if rank<2 else 0
                logical.append(account(ids[:rows],rank));executed.append(account(ids,rank))
                if 'expert_map' in data[rank]:
                    mapping=data[rank]['expert_map'].tolist()
                    expected=[i-rank*32 if rank*32<=i<(rank+1)*32 else -1 for i in range(128)]
                    assert mapping==expected,(rank,mapping)
                    ownership.append({'ep_rank':rank,'map':mapping,'status':'ACTUAL_RUNTIME_MAP_VERIFIED'})
            assert sum(v['valid_paths'] for v in logical)==n*8
            traffic=np.asarray([v['dispatch_bytes_per_destination'] for v in logical])
            recv=traffic.sum(axis=0);send=traffic.sum(axis=1)
            records.append({'model':r['model'],'arm':r['arm'],'size':r['size'],'wave':wave,'layer':layer,
                'prompt_tokens':n,'logical':logical,'executed_or_padded':executed,'runtime_expert_owner_maps':ownership,
                'dispatch_traffic_matrix':traffic.tolist(),'source_send_bytes':send.tolist(),'destination_receive_bytes':recv.tolist(),
                'logical_dispatch_bytes':int(traffic.sum()),'logical_combine_bytes':int(traffic.sum()),
                'max_rank_receive_bytes':int(recv.max()),'mean_rank_receive_bytes':float(recv.mean()),
                'rank_receive_max_mean':float(recv.max()/recv.mean())})
    write(args.output,{'records':records,'labels':'LOGICAL_USEFUL_HIDDEN_BYTES_NOT_WIRE_OR_HBM'})
    print('ACCOUNTING_PASS',len(records))


if __name__=='__main__':main()
