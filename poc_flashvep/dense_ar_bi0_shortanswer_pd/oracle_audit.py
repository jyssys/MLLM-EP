"""Include known feasible subset paths in a larger beam candidate set."""
import copy
import json
from .common import ROOT,write,digest

PROXY='modeled_TTFT_qualified_tokens_per_second'


def rectify(cell):
    value=copy.deepcopy(cell)
    axis=max(('Layer','Token'),key=lambda label:value['oracles'][label]['metrics'][PROXY])
    best=value['oracles'][axis];previous=value['oracles']['All']['metrics'][PROXY]
    changed=previous<best['metrics'][PROXY]
    if changed:
        value['oracles']['All']=copy.deepcopy(best)
        value['oracles']['All']['feasible_subset_fallback']=axis
    mixed=value['oracles']['All']['metrics'][PROXY];denominator=best['metrics'][PROXY]
    value['axis_additional_proxy_headroom']=mixed/denominator-1 if denominator else None
    value['original_All_proxy']=previous
    for key,reference in (('over_hindsight_fixed',value['best_fixed_per_cell_hindsight']),
                           ('over_discovery_frozen_fixed',value['discovery_frozen_fixed'])):
        denominator=reference['metrics'][PROXY]
        value[key]=mixed/denominator if denominator else None
    return value,changed


def main():
    source=ROOT/'analysis/BI0_conditional_axis_oracle_v1.json'
    if not source.exists():return None
    original=json.loads(source.read_text());result=copy.deepcopy(original);changes=[];cells=[]
    for cell in original['cells']:
        value,changed=rectify(cell);cells.append(value)
        if changed:changes.append(dict(QPS=cell['QPS'],seed=cell['seed'],old_proxy=value['original_All_proxy'],
            new_proxy=value['oracles']['All']['metrics'][PROXY],subset=value['oracles']['All']['feasible_subset_fallback']))
    result.update(cells=cells,original_source='analysis/'+source.name,original_SHA256=digest(source),
        correction='The All candidate set includes the measured-model feasible Layer/Token search paths. A finite beam can discard such a path. '
        'This correction does not fit the model, change static selection or create an online result.')
    write(ROOT/'analysis/BI0_conditional_axis_oracle_v2.json',result)
    write(ROOT/'analysis/BI0_oracle_dominance_audit_v1.json',dict(changes=changes,original_SHA256=digest(source),
        evidence='OFFLINE_ORACLE_CORRECTION',scope='Original v1 remains unchanged. '
        'Known subset trajectories prevent a larger candidate set from reporting a negative additional proxy. '
        'This does not make finite-beam results a global bound or validate the queue model.'))
    from .offline_headroom import conditional_validation
    model=json.loads((ROOT/'analysis/BI0_serial_service_model_v2.json').read_text())
    diagnostic=conditional_validation(model,original['decode_step_coefficients_ms'],patterns=('EqualOutputControl_*_BI0_v1',))
    write(ROOT/'analysis/BI0_equal_output_tandem_validation_v1.json',dict(results=diagnostic,
        evidence='OFFLINE_REPLAY_VALIDATION_AGAINST_DIAGNOSTIC_ONLINE',
        scope='No model refit. New arrival seed62 uses exactly the fixed C1 reference output lengths. '
        'This separates unknown Natural EOS from service/queue prediction error. Each GPU arm has one start. '
        'This is not Natural task accuracy, client ITL-tail prediction or online policy performance.'))
    return result


if __name__=='__main__':main()
