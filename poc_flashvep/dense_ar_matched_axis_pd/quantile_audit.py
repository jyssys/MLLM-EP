"""Keep the original CI and audit its discrete-bootstrap quantile convention."""
import json
import math
import shutil
import statistics
import time
from .common import ROOT,digest,write
from .statistics import analysis_stage
from ..dense_ar_layer_token_pd.statistics_report import draws,ratio_interval


def inverse_cdf(values,q):
    assert values and 0<=q<=1
    ordered=sorted(values)
    return ordered[max(0,min(len(ordered)-1,math.ceil(q*len(ordered))-1))]


def cdf_ratio_interval(reference,candidate):
    assert len(reference)==len(candidate)
    if len(reference)<2:return None
    values=[]
    for indices in draws(len(reference)):
        a=statistics.mean(reference[i] for i in indices)
        if not a:return None
        values.append(statistics.mean(candidate[i] for i in indices)/a)
    return [inverse_cdf(values,.025),inverse_cdf(values,.975)]


def main():
    data=ROOT/'analysis/independent_start_statistics.json'
    value=json.loads(data.read_text())
    freeze=ROOT/'state/CI_quantile_sensitivity_design_before_primary_confirmation.json'
    if not freeze.exists():
        primary=sum(r['stage']=='confirmation' for r in value['starts'])
        assert primary==0,'Record a retrospective revision instead of a pre-confirmation freeze'
        snapshot=ROOT/'analysis/independent_start_statistics_type7_before_sensitivity_audit.json'
        assert not snapshot.exists()
        shutil.copyfile(data,snapshot)
        write(freeze,dict(epoch=time.time(),primary_confirmation_starts_seen=primary,
            original_statistics_SHA256=digest(snapshot),
            primary_CI='Unchanged Type7 linear quantile on paired-start bootstrap draws.',
            sensitivity_CI='Inverse empirical CDF on the same draw distribution. Use ceil(q*N)-1 with clipping.',
            decision='Keep both methods. Do not make a strong advantage or noninferiority claim from a quantile-sensitive boundary.',
            observation='One N36 Last12 Output16 cap1 control has a linear upper TTFT ratio0.9995468303958746 and an inverse-CDF upper1.0109602351406097. This is a quantile-definition difference, not a GPU or arithmetic bug.',
            scope='This analysis check follows the observed control boundary. It precedes all primary confirmation starts. No metric, point estimate, SLO, raw evidence, or original CI is replaced. Apply the same sensitivity test to every method and condition.'))
    index={}
    for r in value['starts']:
        stage=analysis_stage(r['stage'])
        index[stage,r['method'],r['layers'],r['cap'],r['QPS'],r['count'],r['admission'],r['seed']]=r
    rows=[]
    for comp in value['comparisons']:
        a=[index[comp['stage'],comp['reference'],comp['layers'] if comp['reference']=='Token' else 0,
            comp['cap'],comp['QPS'],comp['count'],comp['admission'],s] for s in comp['seeds']]
        b=[index[comp['stage'],comp['candidate'],comp['layers'],comp['cap'],comp['QPS'],
            comp['count'],comp['admission'],s] for s in comp['seeds']]
        metrics={}
        for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s'):
            x=[r[key] for r in a];y=[r[key] for r in b]
            original=ratio_interval(x,y)
            assert original==comp['metrics'][key],(comp['reference'],comp['candidate'],key)
            metrics[key]=dict(ratio=original['ratio'],linear_CI95=original['ratio_CI95'],
                empirical_CDF_CI95=cdf_ratio_interval(x,y))
        regimes={}
        for regime in ('TTFT_priority','Balanced','Decode_priority','Common_Balanced_OutputSweep'):
            x=[r['regimes'][regime]['Output_token_SLO_goodput'] for r in a]
            y=[r['regimes'][regime]['Output_token_SLO_goodput'] for r in b]
            original=ratio_interval(x,y)
            assert original==comp['regimes'][regime]['Output_token_SLO_goodput']
            regimes[regime]=dict(ratio=original['ratio'],linear_CI95=original['ratio_CI95'],
                empirical_CDF_CI95=cdf_ratio_interval(x,y))
        relative='NOT_APPLICABLE'
        if comp['reference']=='Native':
            bounds=[metrics[k]['empirical_CDF_CI95'] for k in ('ITL_p99_ms','TPOT_p95_ms')]
            relative='PASS' if comp['starts']>=3 and all(c is not None and c[1]<=1.05 for c in bounds) else 'NOT_ESTABLISHED'
        rows.append(dict(**{k:comp[k] for k in ('stage','reference','candidate','layers','cap','QPS','count','admission','starts','seeds')},
            metrics=metrics,SLO_goodput_ratios=regimes,Native_relative_empirical_CDF_5pct=relative,
            original_Native_relative_5pct=comp['Native_relative_5pct']))
    write(ROOT/'analysis/CI_quantile_sensitivity.json',dict(epoch=time.time(),comparisons=rows,
        evidence='TRACE_ANALYSIS',original_CIs_preserved=True,
        source='https://numpy.org/doc/stable/reference/generated/numpy.quantile.html',
        scope='Two documented quantile conventions on identical paired-start bootstrap draws. For n<=6 all n^n ordered index tuples are enumerated. This is not exact population coverage. Sparse starts remain a statistical limitation. No request/token pseudoreplication.'))
    print('QUANTILE_SENSITIVITY_AUDIT',len(rows),flush=True)


if __name__=='__main__':
    main()
