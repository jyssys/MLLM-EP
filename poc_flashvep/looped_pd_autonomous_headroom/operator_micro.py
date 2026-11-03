"""Actual-checkpoint differential projection test. Not a serving speedup."""
import statistics
from .common import *


def main():
    activate()
    cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text());folder=ROOT/'raw'/cfg['name']
    import torch
    torch.set_grad_enabled(False);torch.cuda.set_device(0);torch.manual_seed(cfg['seed'])
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT
    os.environ.update(B_NUMERICAL_VARIANT='down_norm' if cfg['model']=='ouro' else 'down_norm_adapter',B_NUMERICAL_RUN=cfg['name'])
    from ..looped_pd_b_numerical_diagnosis.patches import install
    install()
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
    from .tile_graph import GroupedProjection
    results=[]
    for name,module in model.named_modules():
        if not isinstance(module,torch.nn.Linear):continue
        if not(name.endswith('down_proj') or '.mlp.proj' in name or cfg['model']=='huginn' and name.endswith('adapter')):continue
        candidate=GroupedProjection(module.forward,name,cfg['projection_graph_rows'])
        for rows in cfg['operator_rows']:
            x=torch.randn((rows,module.in_features),device='cuda:0',dtype=torch.bfloat16)
            expected=module(x);actual=candidate(x);torch.cuda.synchronize()
            exact=torch.equal(expected,actual)
            delta=(expected.float()-actual.float())
            row=dict(name=name,rows=rows,columns=module.in_features,output_columns=module.out_features,
                exact=exact,exact_element_fraction=(expected==actual).float().mean().item(),
                maximum_absolute_error=delta.abs().max().item(),relative_L2=(delta.norm()/expected.float().norm().clamp_min(1e-30)).item(),
                evidence='MEASURED_ACTUAL_CHECKPOINT_OPERATOR_NOT_E2E')
            if rows in (cfg['projection_graph_rows'],8192):
                for function,label in ((module,'baseline'),(candidate,'candidate')):
                    durations=[]
                    for _ in range(5):
                        left=torch.cuda.Event(enable_timing=True);right=torch.cuda.Event(enable_timing=True)
                        left.record();function(x);right.record();right.synchronize();durations.append(left.elapsed_time(right))
                    row[label+'_device_ms_median']=statistics.median(durations)
            results.append(row);append(folder/'operator_results.jsonl',row)
            assert exact,'Projection graph changed BF16 values: '+name
    write(folder/'operator_summary.json',dict(all_exact=all(r['exact'] for r in results),comparisons=len(results),
        model=cfg['model'],peak_HBM_bytes=torch.cuda.max_memory_allocated(),
        scope='All selected actual-weight projections. This test does not establish model output parity or serving latency.'))
    write(folder/'status.json',dict(status='COMPLETE'))


if __name__=='__main__':main()
