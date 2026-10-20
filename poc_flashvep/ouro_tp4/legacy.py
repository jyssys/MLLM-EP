"""Re-run legacy workers without writing any previous experiment artifact."""
import argparse
import gzip
import queue
import time
import torch
from poc_flashvep.ouro_request_pp.runtime import Pipeline
from .contract import *


class FrozenQueue(queue.Queue):
    def configure(self,engine,requests):
        terminal=engine.path[-1][0] if engine.path else 3
        self.feedback={r['id']:torch.tensor(r['teacher_ids'],device=f'cuda:{terminal}').view(-1,1,1) for r in requests}
        self.cpu={r['id']:r['teacher_ids'] for r in requests};self.computed=[]
    def put(self,value,*args,**kwargs):
        if value[0]=='token':
            _,item,token,actual,finish=value;self.computed.append((item.rid,item.step,actual))
            value=('token',item,self.feedback[item.rid][item.step],self.cpu[item.rid][item.step],finish)
        return super().put(value,*args,**kwargs)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['layer_pp4','loop_pp4'],required=True)
    p.add_argument('--indices',default='0,1');p.add_argument('--label',required=True);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--locked-greedy',action='store_true');p.add_argument('--trace',action='store_true');a=p.parse_args()
    mask();torch.set_num_threads(1);engine=Pipeline(a.arm)
    if not a.locked_greedy:engine.completed=FrozenQueue()
    selected=set(map(int,a.indices.split(',')));lastshape=None
    for c in cases():
        if c['index'] not in selected:continue
        requests=c['requests'];O=c['O'];shape=(tuple(sorted({len(r['input_ids']) for r in requests})),O)
        if shape!=lastshape:
            for part in engine.parts:part.graphs.clear();part.caches.clear()
            lastshape=shape
        start=time.perf_counter()
        if not a.locked_greedy:
            # The terminal rank is GPU7 for fixed-R4 in both placements.
            engine.path=[(3,3,0,24)];engine.completed.configure(engine,requests)
        for P in sorted({len(r['input_ids']) for r in requests}):
            row=next(r for r in requests if len(r['input_ids'])==P)
            engine.run([row],4,O,graphs=True)
        prep=time.perf_counter()-start
        if not a.locked_greedy:engine.completed.configure(engine,requests)
        result=engine.run(requests,4,O,graphs=True,trace=a.trace)
        if a.locked_greedy:
            result['locked_output_agreement']=all(out['ids']==r['teacher_ids'] for out,r in zip(result['outputs'],requests))
        else:result['computed_argmax']=engine.completed.computed
        result.update(arm=('LPP4_legacy' if a.arm=='layer_pp4' else 'LoopPP4_legacy'),case={k:v for k,v in c.items() if k!='requests'},
            repeat=a.repeat,preparation_wall_s=prep,teacher_forced=not a.locked_greedy)
        target=ROOT/'raw'/a.label;target.mkdir(parents=True,exist_ok=True)
        file=target/f'case_{c["index"]:02d}.json.gz';assert not file.exists()
        with gzip.open(file,'wt') as f:json.dump(result,f,separators=(',',':'))
        print('LEGACY',a.arm,c['index'],round(result['wall_s'],4),flush=True)
    engine.close()


if __name__=='__main__':main()
