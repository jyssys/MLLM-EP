"""Inspect two existing C3 cache keys without changing cached artifacts."""
import json
from pathlib import Path
from .common import ROOT,digest,write


def main():
    base=Path('/home/esjung/.cache/vllm/torch_compile_cache')
    paths={label:base/key/'rank_0_0/backbone' for label,key in
           (('coalesced','da7f3c4039'),('no_coalesce_synchronized','865285c863'))}
    factors={label:json.loads((path/'cache_key_factors.json').read_text()) for label,path in paths.items()}
    a,b=factors.values()
    different=[key for key in sorted(a.keys()|b.keys()) if a.get(key)!=b.get(key)]
    env_a=a.get('env',{});env_b=b.get('env',{})
    env_different=[key for key in sorted(env_a.keys()|env_b.keys()) if env_a.get(key)!=env_b.get(key)]
    hashes={label:{name:digest(path/name) for name in ('computation_graph.py',
        'artifact_compile_range_1_8192_subgraph_0','artifact_compile_range_1_8192_subgraph_1',
        'artifact_compile_range_1_8192_subgraph_36')} for label,path in paths.items()}
    result=dict(paths={key:str(value) for key,value in paths.items()},different_top_level_keys=different,
        different_environment_keys=env_different,hashes=hashes,
        computation_graph_equal=len({value['computation_graph.py'] for value in hashes.values()})==1,
        evidence='TRACE_ANALYSIS',scope='Read-only comparison of existing cache artifacts. '
        'The side-channel port changes the cache namespace. Equal generated computation graphs do not prove equal selected kernels. '
        'Different serialized artifacts do not prove different numerical kernels. This does not identify the producer tensor mismatch cause.')
    write(ROOT/'analysis/BI0_coalescing_compile_cache_key_audit_v1.json',result)
    return result


if __name__=='__main__':main()
