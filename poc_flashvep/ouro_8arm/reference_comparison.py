"""Compare actual CDB and prototype outputs without conflating kernel backends."""
from .contract import *


def compare_reference(reference,candidate):
    rows=[]
    for file in sorted((ROOT/'raw'/reference).glob('*.json')):
        a=read(file)
        if not isinstance(a,dict) or 'output_ids' not in a:continue
        b=read(ROOT/'raw'/candidate/file.name)
        assert a['prompt_ids']==b['prompt_ids']
        rows.append(dict(id=a['id'],reference=reference,candidate=candidate,prompt_token_exact=True,
            reference_tokens=a['output_ids'],candidate_tokens=b['output_ids'],
            reference_depths=a['depths'],candidate_depths=b['depths'],
            output_exact=a['output_ids']==b['output_ids'],depth_trace_exact=a['depths']==b['depths'],
            scope='Same checkpoint, gate, threshold, and last_exited policy. FA3/paged CDB versus masked-capacity SDPA prototype.'))
    assert len(rows)==4
    write(ROOT/'numerical/dynamic_reference_comparison.json',dict(rows=rows,
        status='BOUNDED_OUTPUT_AND_DEPTH_MATCH' if all(r['output_exact'] and r['depth_trace_exact'] for r in rows) else 'BACKEND_OUTPUT_OR_GATE_DRIFT_UNRESOLVED',
        limit='This bounded probe does not establish full benchmark accuracy or cross-backend bitwise KV equality.'))
    print('DYNAMIC_REFERENCE',sum(r['output_exact'] for r in rows),'outputs,',sum(r['depth_trace_exact'] for r in rows),'depth traces, of',len(rows),flush=True)


if __name__=='__main__':
    import sys
    compare_reference(*sys.argv[1:])
