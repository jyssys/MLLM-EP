"""Recalculate campaign analyses without starting a GPU experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,preserve,bind


def command(module,*arguments):
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    reserved={cpu for cpu in os.sched_getaffinity(0) if cpu>=160}
    return subprocess.run([str(PYTHON),'-B','-m','poc_flashvep.dense_ar_p_critical_path.'+module,*arguments],
        cwd=REPO,check=True,env=environment,
        preexec_fn=(lambda:os.sched_setaffinity(0,reserved)) if reserved else None)


def analyses():
    bind();preserve()
    command('historical_recalculation');command('critical_analysis','analyze');command('causal_summary');command('serial_analysis')
    plans=(('c3_credit_retirement_confirmation_r1.json','c3_credit_confirmation'),
        ('p_geometry_confirmation_v1.json','p_geometry_confirmation'),
        ('p_output_stress_c3_v1.json','p_output_stress'),
        ('coordinator_hygiene_c3_confirmation_stable_stock_v5.json','coordinator_hygiene_stable_stock_confirmation'),
        ('coordinator_hygiene_c3_confirmation_fixed_ports_v6.json','coordinator_hygiene_fixed_ports_confirmation'),
        ('coordinator_hygiene_c3_confirmation_warm_regime_v8.json','coordinator_hygiene_warm_regime_confirmation'),
        ('coordinator_hygiene_c3_confirmation_warm_regime_v9.json','coordinator_hygiene_warm_regime_confirmation'),
        ('coordinator_hygiene_c3_confirmation_warm_regime_v10.json','coordinator_hygiene_warm_regime_confirmation'),
        ('output192_D_cap_intervention_v1.json','output192_D_cap_intervention'),
        ('p_geometry_D_cap4_confirmation_v1.json','p_geometry_D_cap4_confirmation'))
    for plan,stem in plans:
        if (ROOT/'jobs'/plan).exists():command('confirmation_analysis','--plan',plan,'--stem',stem)
    if (ROOT/'state/p_geometry_D_cap4_selection_frozen.json').exists():command('cap_intervention','discovery-analysis')
    if (ROOT/'jobs/output192_D_cap_intervention_v1.json').exists():command('output_admission_control','realization')
    if (ROOT/'state/fixed_port_stock_strict_complete_v6_reader_fix1.json').exists():command('cache_regime_analysis')
    command('arrivals');command('accuracy_analysis');command('verdict')
    if (ROOT/'attention_poc/configs/attention_poc_v1.json').exists():command('attention_analysis')
    preserve()


def figure_check():
    manifest=ROOT/'analysis/figure_manifest.json';assert manifest.exists()
    expected=json.loads(manifest.read_text())
    before={entry['path']:entry['SHA256'] for row in expected['figures'] for entry in row['files'].values()}
    assert all(digest(ROOT/path)==sha for path,sha in before.items()),'A saved figure or table was modified'
    command('figures')
    current=json.loads(manifest.read_text())
    after={entry['path']:entry['SHA256'] for row in current['figures'] for entry in row['files'].values()}
    equal=before==after
    result=dict(time=time.time(),passed=equal,files=len(after),before=before,after=after,
        scope='Regenerate every registered PNG,PDF and exact source CSV from the same analysis files. '
            'Compare bytes,not only filenames. This does not validate the physical measurement scope or replace visual inspection.')
    write(ROOT/'analysis/figure_regeneration_audit.json',result);assert equal,result
    print('FIGURE_BYTE_REGENERATION_PASS',len(after),flush=True)
    return result


def dataset(archive,output):
    bind()
    from ..dense_ar_output_aware_v2.dataset_provenance import reconstruct
    return reconstruct(Path(archive),Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('analyze','figures','verify-figures','dataset'))
    parser.add_argument('--archive');parser.add_argument('--output');args=parser.parse_args()
    if args.action=='analyze':analyses()
    elif args.action=='figures':command('figures')
    elif args.action=='verify-figures':figure_check()
    else:
        assert args.archive and args.output,'Dataset reconstruction needs the pinned source archive and a new output directory'
        dataset(args.archive,args.output)
