"""Offline fixed-regime composition. This is not a dynamic serving oracle."""
import csv
import json
import statistics
from .common import ROOT,write
from .analyze import csv_write

REGIMES=('P_pressure','balanced','verified_D_pressure')


def main():
    rows=[r for r in csv.DictReader((ROOT/'static_cut_metrics_by_start.csv').open())
          if int(r['prompt_tokens'])==512 and r['handoff_mode']=='cut_snapshot']
    dest=ROOT/'configs/frozen_regime_selections.json'
    if not dest.exists():
        selections={}
        for model in ('ouro','huginn'):
            calibration=[r for r in rows if r['model']==model and int(r['start'])==0 and r['regime'] in REGIMES]
            costs={(r['regime'],int(r['cut'])):float(r['all_requests_e2e_ms_mean']) for r in calibration}
            cuts=sorted({k[1] for k in costs})
            if not all((regime,cut) in costs for regime in REGIMES for cut in cuts):
                raise RuntimeError('Calibration map is incomplete')
            static=min(cuts,key=lambda c:sum(costs[(g,c)] for g in REGIMES))
            conditioned={g:min(cuts,key=lambda c:costs[(g,c)]) for g in REGIMES}
            selections[model]=dict(static_cut=static,regime_cuts=conditioned,
                                   calibration_static_mean_ms=statistics.mean(costs[(g,static)] for g in REGIMES),
                                   calibration_selected_mean_ms=statistics.mean(costs[(g,conditioned[g])] for g in REGIMES))
        write(dest,dict(selections=selections,
              rule='Choose minimum mean all-request E2E on start0, with equal weights for three homogeneous profiles.',
              evidence='OFFLINE_FIXED_REGIME_COMPOSITION_NOT_SERVING_ORACLE',
              limits='No live switching, no per-request counterfactual, no stable-rate or goodput bound. Regime labels are external experimental conditions.',
              registration='Deterministic start0-only selection. Later repeats were already in progress. Not a preregistered main SLO metric.'))
    selections=json.loads(dest.read_text())['selections'];output=[]
    for model,selected in selections.items():
        for repeat in sorted({int(r['start']) for r in rows if r['model']==model}):
            costs={(r['regime'],int(r['cut'])):float(r['all_requests_e2e_ms_mean']) for r in rows
                   if r['model']==model and int(r['start'])==repeat and r['regime'] in REGIMES}
            static=selected['static_cut'];choices=selected['regime_cuts']
            if not all((g,static) in costs and (g,choices[g]) in costs for g in REGIMES):continue
            base=statistics.mean(costs[(g,static)] for g in REGIMES)
            chosen=statistics.mean(costs[(g,choices[g])] for g in REGIMES)
            output.append(dict(model=model,start=repeat,calibration=repeat==0,static_cut=static,
                               selected_cuts=json.dumps(choices,sort_keys=True),static_mean_E2E_ms=base,
                               selected_mean_E2E_ms=chosen,latency_reduction_pct=100*(1-chosen/base),
                               evidence='OFFLINE_FIXED_REGIME_COMPOSITION',
                               sustainable_goodput_gain='NOT_MEASURED',dynamic_oracle='NOT_RUN'))
    csv_write('fixed_regime_selection_comparison.csv',output)
    print(json.dumps(selections,indent=2))


if __name__=='__main__':main()
