"""Restrict every same-window gate to the recorded device capture interval."""
from .common import *


def one(cfg):
    activate()
    from ..looped_pd_headroom_recovery.gates import one as inherited
    path = ROOT/'raw'/cfg['name']/'profile_analysis.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('status') == 'PARSED' and not value.get('bounded_capture'):
            prior = path.with_name('profile_analysis_before_alignment.json')
            if not prior.exists():
                write(prior, value)
            value['bounded_capture'] = True
            value['alignment_rule'] = 'Intersect the serving interval with the actual capture. Unrecorded time is not GPU idle.'
            write(path, value)
    return inherited(cfg)


def refresh_selection():
    path = ROOT/'state/adaptive_search_results.json'
    if not path.exists():
        return
    prior = ROOT/'state/adaptive_search_results_before_alignment.json'
    records = json.loads(path.read_text())
    if not prior.exists():
        write(prior, records)
    selected = []
    for record in records:
        if record.get('complete') and record.get('profile', {}).get('status') == 'PARSED':
            cfg = json.loads((ROOT/'configs'/(record['profile_name']+'.json')).read_text())
            record['gates'] = one(cfg)
            record['profile'] = json.loads((ROOT/'raw'/cfg['name']/'profile_analysis.json').read_text())
    for model in ('ouro', 'huginn'):
        candidates = [r for r in records if r.get('gates', {}).get('model') == model]
        if not candidates:
            continue
        def score(row):
            g = row['gates']
            return (bool(g.get('P_compute_gate') and g.get('D_SLO_gate') and g.get('D_credit_gate')),
                    bool(row['profile'].get('eligible_for_matched_clean')), bool(g.get('D_SLO_gate')),
                    g.get('P_actual_kernel_busy_central') or 0,
                    -(g.get('D_credit_refusal_sample_fraction') or 0))
        chosen = max(candidates, key=score)
        cfg = json.loads((ROOT/'configs'/(chosen['name']+'.json')).read_text())
        selected.append(dict(configuration=cfg, evidence=chosen,
                             selection_scope='Best retained diagnostic candidate after capture-window alignment. Not a success verdict.'))
    write(ROOT/'state/adaptive_search_results_aligned.json', records)
    old = ROOT/'state/adaptive_selection.json'
    if old.exists() and not (ROOT/'state/adaptive_selection_before_alignment.json').exists():
        write(ROOT/'state/adaptive_selection_before_alignment.json', json.loads(old.read_text()))
    write(old, selected)


if __name__ == '__main__':
    refresh_selection()
