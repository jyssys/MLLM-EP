"""Register a short teardown/callback smoke before full-answer quality work."""
import json
from .common import ROOT,PRIOR,write,digest

def main():
    assert not (ROOT/'p4_quality_smoke_plan.json').exists()
    selected=json.loads((ROOT/'objective_shortlist.json').read_text())['primary_objective']
    joint=dict(selected,optimizer=selected['label'])
    cost=PRIOR/'cost_model_artifacts/service_lookup_v1.json'
    questions=json.loads((ROOT/'quality_manifest.json').read_text())['rows'][:2]
    job=dict(id='dual_p4_SMOKE_gpu4_v1',kind='dual_quality',gpu=4,
        cells=[dict(label='F0A',runtime='atomic',C=1024,fairness='stock'),joint],
        ids=[r['id'] for r in questions],width=2,cap=256,compute_budget_ms=120.,budget=2048,blocks=18432,
        cost_model=str(cost),cost_model_sha256=digest(cost),timeout_s=1800,
        scope='Callback, teardown, and Full-R4 smoke only. A256-token cap is not an accuracy evaluation.')
    write(ROOT/'p4_quality_smoke_plan.json',dict(gpus=[4],jobs=[job]))
    write(ROOT/'execution_dependencies.json',{'p4_quality_plan.json':['p4_quality_smoke_plan.json']})
    print('Registered quality preflight without changing frozen benchmark settings')

if __name__=='__main__':main()
