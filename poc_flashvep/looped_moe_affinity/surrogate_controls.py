"""Quantify sparse-table noise in the degree-preserving co-activation null."""
import csv
import json
import numpy as np
from .common import ROOT
from .affinity import slot_null,coactivation,tv
from .aggregate_quality import csv_write

def main():
    path=ROOT/'coactivation_matrix_summary.csv'
    output=list(csv.DictReader(path.open()))
    rng=np.random.default_rng(20261027)
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        archive=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        manifest=json.loads((ROOT/'raw'/f'{model}_route_arrays_manifest.json').read_text())
        lookup={tuple(v['key']):v['index'] for v in manifest}
        for row in output:
            if row['model']!=model:continue
            key=(int(row['batch']),int(row['context']),row['domain'],row['site'],int(row['loop']),'heldout')
            ids=archive[f'ids_{lookup[key]}'];e=25 if model.startswith('ifm') else (64 if row['recurrent']=='True' else 8)
            null=np.stack([coactivation(slot_null(ids,rng),e) for _ in range(12)])
            mean=null.mean(0);observed=coactivation(ids,e)
            expected=[tv(v+.01,(null.sum(0)-v)/11+.01) for v in null]
            actual=tv(observed+.01,mean+.01)
            row.update(pair_tv_from_slot_null=actual,null_leave_one_out_tv_mean=float(np.mean(expected)),
                null_leave_one_out_tv_p95=float(np.quantile(expected,.95)),excess_tv_over_surrogate_noise=actual-np.mean(expected),
                significance_warning='Descriptive excess over matched sparse-table noise. No multiple-comparison significance claim.')
    csv_write(path,output)

if __name__=='__main__':main()
