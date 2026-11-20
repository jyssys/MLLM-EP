"""Rebuild derived analyses after all owned GPU jobs finish."""
import json
import time
from .common import ROOT,write,preserve


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    from .axis_timeline import analyze
    from .serving_analysis import analyze as serving_analyze
    outcomes=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        cfg=json.loads((folder/'configuration.json').read_text())
        if not cfg.get('controlled_pipeline') and not folder.name.startswith('ContinuationCost_'):continue
        if not all((folder/file).exists() for file in ('requests.jsonl','handoffs.jsonl',
                'actual_queries_prefill.json','actual_queries_decode.json')):continue
        if (folder/'window.json').exists():serving_analyze(folder.name)
        value=analyze(folder.name)
        assert value['all_prompt_work_accounted'],folder.name
        outcomes.append(dict(name=folder.name,N=len(value['rows']),prompt_coverage=True))
    from .component_analysis import main as components
    from .online_decomposition import main as decomposition
    from .confirmation_analysis import main as confirmation
    from .main_table import main as table
    from .pipeline_queues import main as pipeline_queues
    from .delivery_tails import main as delivery_tails
    from .figures import main as figures
    from .serving_figures import main as serving_figures
    from .oracle_audit import main as oracle_audit
    from .load_service_audit import main as load_service_audit
    from .metric_reproduction_audit import main as metric_audit
    metric_audit();components();decomposition();confirmation(fast=True);table();pipeline_queues();delivery_tails();oracle_audit();load_service_audit();figures();serving_figures()
    from .experiment_register import main as register
    register()
    preserve()
    write(ROOT/'analysis/final_CPU_reanalysis_receipt.json',dict(time=time.time(),jobs=outcomes,
        scope='Derived analyses use the final parser. Frozen raw evidence is unchanged.'))


if __name__=='__main__':main()
