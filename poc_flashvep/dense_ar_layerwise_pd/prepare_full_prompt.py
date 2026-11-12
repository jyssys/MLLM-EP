"""Require exact GPU Hidden transport before full-prompt Layer cuts."""
import json
from .common import *
from .prepare import base,save


def main():
    initialize()
    gate=json.loads((ROOT/'analysis/hidden_nixl_N32_qualification.json').read_text());assert gate['passed']
    jobs=[]
    for count in (8,32):
        modes=[('STATE_REFERENCE',36)]
        modes += [('MIGRATE',c) for c in (35,34,32,28)]
        for mode,cut in modes:
            name=f'layer_qwen8_L8000_O16_{mode.lower()}_c{cut}_N{count}_gpu_v6'
            c=base(name,count=count,cap=16)
            c.update(layer_mode=mode,layer_cut=cut if cut<36 else 32,state_capture=True,
                partial_KV_wire=mode=='MIGRATE',serialize_D_imports=True,
                hidden_transport='NIXL_GPU',hidden_source_slots=16,hidden_destination_slots=4,
                hidden_verify_hash=True,save_state_tensors=False,
                D_admission_batch=4,D_cohort_collect_ms=2,cohort_barrier=True,
                warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,
                fixed_KV_bytes=42*1024**3,isolate_layer_compile_cache=True,
                timeout=360,wall_timeout=900,calibration_serial=count==8,
                phase='FULL_PROMPT_LAYER_STATE_QUALIFICATION')
            jobs.append(save(c))
    write(ROOT/'state/full_prompt_correctness_plan.json',dict(jobs=jobs,
        scope='Full 8000-token Prefix/KV/Hidden exactness, then N32 simultaneous submission. State capture and payload hashing are diagnostic and excluded from clean performance.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
