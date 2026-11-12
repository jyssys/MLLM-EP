"""Use a global chunk budget, not only a per-request threshold."""
import json
from .common import *
from .prepare import base,save


def main():
    initialize();jobs=[]
    for count in (8,32):
        for chunk in (256,512,1024,2048):
            c=base(f'colocated_qwen8_L8000_O16_chunk{chunk}_N{count}_qual_v1',count=count,cap=16)
            c.update(arm='DP2',chunk=chunk,token_budget=chunk,atomic_prompt=False,
                fixed_KV_bytes=42*1024**3,warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,
                batch_commands=True,phase='COLOCATED_NATIVE_CHUNK_CORRECTNESS',timeout=360)
            jobs.append(save(c))
    write(ROOT/'state/colocated_qualification_plan.json',dict(jobs=jobs,
        scope='Two full-model local replicas on the same two GPUs. The total token budget is the chunk size, so a large mixed Prefill batch cannot bypass the intended quantum. This is not Token-suffix P-to-D migration.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
