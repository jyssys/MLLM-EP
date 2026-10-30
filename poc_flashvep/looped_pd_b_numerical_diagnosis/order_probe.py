"""Reorder diagnostic requests while preserving each request's sampling seed."""
import os
from dataclasses import replace


def install(module):
    if os.environ.get('B_NUMERICAL_PRESERVE_SOURCE_SEEDS')!='1':return
    original=module.serve
    def serve(engine,inputs,args,run,warmup=False):
        add=engine.add_request
        append=module.append
        def add_request(rid,tokens,params):
            index=int(rid.split('_q')[1])
            source=module.DATA[index]['source_sample_id']
            return add(rid,tokens,replace(params,seed=module.CONFIG['seed']+source))
        def append_record(path,value):
            if path.name=='requests.jsonl' and 'sample_id' in value:
                source=module.DATA[value['sample_id']]['source_sample_id']
                value=dict(value,source_sample_id=source,seed=module.CONFIG['seed']+source,
                           order_test='Source request seeds remain fixed across admission permutations')
            return append(path,value)
        engine.add_request=add_request;module.append=append_record
        try:return original(engine,inputs,args,run,warmup=warmup)
        finally:engine.add_request=add;module.append=append
    module.serve=serve


def main():
    import json
    from .common import ROOT,PRIOR,write
    data=json.loads((PRIOR/'data/W1.json').read_text())
    reversed_rows=[dict(data[i],source_sample_id=i) for i in reversed(range(8))]
    write(ROOT/'data/W1_reversed8.json',reversed_rows)


if __name__=='__main__':main()
