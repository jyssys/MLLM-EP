"""Quantify the non-bitwise hidden groups from the stored rank audit."""
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    audit=read(ROOT/'results/TP_replica_state_audit.json')
    groups=[r for r in audit['records'] if r['field']=='hidden' and not r['all_ranks_bitwise_equal']]
    records=[]
    for row in groups:
        directory=ROOT/f'results/native_fidelity_TP{row["TP"]}_v1'
        stem=f'teacher_{file_id(row["id"])}.step{row["step"]}'
        arrays=[]
        for rank in range(row['TP']):
            with np.load(directory/f'{stem}.rank{rank}.npz') as value:
                arrays.append(value['hidden'].astype(np.float64))
        relative=[float(np.linalg.norm(value-arrays[0])/max(np.linalg.norm(arrays[0]),1e-30)) for value in arrays[1:]]
        records.append(dict(TP=row['TP'],id=row['id'],step=row['step'],
            maximum_absolute_difference=row['maximum_absolute_rank_difference'],
            maximum_relative_L2=max(relative)))
    target=ROOT/'results/TP_hidden_rank_drift_summary.json'
    assert not target.exists()
    write(target,dict(non_bitwise_groups=len(records),all_group_count=120,records=records,
        maximum_absolute_difference=max((r['maximum_absolute_difference'] for r in records),default=0.),
        maximum_relative_L2=max((r['maximum_relative_L2'] for r in records),default=0.),
        scope='Cross-rank hidden comparison on stored teacher states. Not same-rank repeatability or an independent quality result.',
        causal_operator='NOT_CAPTURED. Do not attribute the difference to a specific collective or kernel.'))
    print(dict(non_bitwise=len(records),maximum_relative_L2=max(r['maximum_relative_L2'] for r in records),
               maximum_absolute_difference=max(r['maximum_absolute_difference'] for r in records)))


if __name__=='__main__':
    main()
