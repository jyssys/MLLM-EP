"""Check replicated arrays without treating TP ranks as independent problems."""
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    inputs=read(ROOT/'results/fidelity_inputs.json')+read(ROOT/'results/fidelity_additional_inputs.json')
    records=[]
    for tp in [2,4]:
        directory=ROOT/f'results/native_fidelity_TP{tp}_v1'
        for row in inputs:
            for step in [0,1,4,32,128,1024]:
                paths=[directory/f'teacher_{file_id(row["id"])}.step{step}.rank{rank}.npz' for rank in range(tp)]
                values=[np.load(path) for path in paths]
                for field in ['logits','hidden']:
                    arrays=[value[field] for value in values]
                    records.append(dict(TP=tp,id=row['id'],step=step,field=field,
                        all_ranks_bitwise_equal=all(np.array_equal(arrays[0],v) for v in arrays[1:]),
                        maximum_absolute_rank_difference=max(float(np.max(np.abs(arrays[0]-v))) for v in arrays[1:])))
                for value in values:
                    value.close()
    target=ROOT/'results/TP_replica_state_audit.json'
    assert not target.exists()
    write(target,dict(records=records,all_logits_replicate_bitwise=all(r['all_ranks_bitwise_equal'] for r in records if r['field']=='logits'),
        all_hidden_replicate_bitwise=all(r['all_ranks_bitwise_equal'] for r in records if r['field']=='hidden'),
        independent_teacher_problems=len(inputs),
        scope='Reuse existing selected teacher states. TP ranks and checkpoints are not independent serving samples.',
        KV='Not compared across ranks because local head ownership differs.'))
    print({field:sum(r['all_ranks_bitwise_equal'] for r in records if r['field']==field) for field in ['logits','hidden']})


if __name__=='__main__':
    main()
