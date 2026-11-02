"""Generic exact descriptor construction. No KV value or transfer rule changes."""
import itertools


def valid_segments(info,tables,start,end,recurrent_layers,coda_layers,prompt_length):
    recurrent=set(recurrent_layers)
    coda=set(coda_layers)
    block_size=info['block_size']
    used=[block for table in tables for block in table]
    if len(set(used))!=len(used):
        raise ValueError('Packed native descriptors require non-aliased depth block tables')
    dense_layer=info['layer_bytes']==block_size*info['token_bytes']
    for depth,table in enumerate(tables):
        active=[layer for layer in range(info['num_layers']) if not depth or layer in recurrent]
        position=start
        while position<end:
            page=position//block_size
            offset=position%block_size
            page_end=min(end,(page+1)*block_size)
            segments=[]
            for layer in active:
                limit=min(page_end,prompt_length-1) if layer in coda else page_end
                count=max(0,limit-position)
                if not count:continue
                # Only complete adjacent layer pages can merge. Partial rows
                # retain the legacy addresses and exact valid-token lengths.
                if dense_layer and offset==0 and count==block_size and segments and segments[-1][2]==block_size and segments[-1][0]+segments[-1][1]==layer:
                    previous=segments[-1]
                    segments[-1]=(previous[0],previous[1]+1,count)
                else:
                    segments.append((layer,1,count))
            for layer,run,count in segments:
                size=count*info['token_bytes'] if run==1 else run*info['layer_bytes']
                for base in (info['key_ptr'],info['value_ptr']):
                    yield (base+table[page]*info['block_bytes']+layer*info['layer_bytes']+offset*info['token_bytes'],size,info['device'])
            position=page_end


def replay(value,function):
    from ..looped_pd_abc.segments import coalesce
    a=list(function(value['source_info'],value['source_tables'],value['start'],value['end'],
        value['recurrent_layers'],value['coda_layers'],value['prompt_tokens']))
    b=list(function(value['target_info'],value['target_tables'],value['start'],value['end'],
        value['recurrent_layers'],value['coda_layers'],value['prompt_tokens']))
    if value.get('final'):
        s,t=value['source_info'],value['target_info']
        a.append((s['hidden_ptr']+value['source_slot']*s['hidden_bytes'],s['hidden_bytes'],s['device']))
        b.append((t['hidden_ptr']+value['target_slot']*t['hidden_bytes'],t['hidden_bytes'],t['device']))
    merged=coalesce(a,b)
    return merged,len(a)


def benchmark(path):
    import json,time,statistics,sys
    from .common import RLT
    sys.path.insert(0,str(RLT))
    from ..looped_pd_abc.boundary import valid_segments as legacy
    from .common import ROOT,write
    value=json.loads(path.read_text())
    results=[]
    outputs={}
    for label,function in [('legacy',legacy),('packed',valid_segments)]:
        timings=[]
        for _ in range(5):
            begin=time.perf_counter()
            result,n=replay(value,function)
            timings.append(time.perf_counter()-begin)
        outputs[label]=result
        results.append(dict(method=label,input_descriptors=n,coalesced_descriptors=len(result[0]),
            median_CPU_ms=1000*statistics.median(timings),all_CPU_ms=[1000*x for x in timings]))
    output=dict(replay=str(path),descriptor_pairs_exact=outputs['legacy']==outputs['packed'],results=results,
        evidence='ACTUAL_NATIVE_METADATA_CPU_REPLAY',scope='No GPU memory is read. CPU replay timings are not E2E GPU speedup.')
    write(ROOT/'state/descriptor_cpu_replay.json',output)
    assert output['descriptor_pairs_exact']
    return output


if __name__=='__main__':
    import sys,json
    from pathlib import Path
    print(json.dumps(benchmark(Path(sys.argv[1]))))
