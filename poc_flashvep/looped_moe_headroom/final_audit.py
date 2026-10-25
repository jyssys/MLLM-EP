"""Validate report evidence and a fresh archive extraction. No GPU is used."""
import csv
import json
from pathlib import Path
import tarfile
import tempfile

from .common import ROOT,MODELS,SPEC,digest,write

def main():
    checks={}
    assert json.loads((ROOT/'validation.json').read_text())['status']=='PASS'
    checks['live_evidence_integrity']='PASS'
    pins=json.loads((ROOT/'model_pins.json').read_text())
    source=json.loads((ROOT/'source_lock.json').read_text())
    assert pins==source['models']
    assert digest(SPEC)==source['spec_sha256']
    checks['source_and_model_pins']='PASS'
    decisions=list(csv.DictReader((ROOT/'decision_summary.csv').open()))
    assert {r['model'] for r in decisions}==set(MODELS)
    for r in decisions:
        expected=1/(1-float(r['f_expert_clean_denominator_optimistic'])*float(r['effective_q']))
        assert abs(expected-float(r['optimistic_proportional_speedup']))<1e-12
        assert r['measured_candidate_speedup']=='' and r['downstream_quality_margin']=='NOT_ESTABLISHED'
        assert r['p3']=='NOT_RUN_GATE_NOT_PASSED'
    checks['three_model_report_estimates_not_measured_speedups']='PASS'
    manifest=json.loads((ROOT/'archive_manifest.json').read_text())
    assert manifest['lossless_sha256_verified'] and manifest['originals_retained']
    archives=manifest['archives']
    for r in archives:
        assert digest(ROOT/'archives'/r['archive'])==r['sha256']
    chosen=[r for r in archives if r['archive'] in [
        'ifm_ouro_heavy_profile_v1.tar.xz','foil1_strict_load_final_v1.tar.xz',
        'foil1_independent_control_shard0_v1.tar.xz','logs.tar.xz']]
    assert len(chosen)==4
    restored=0
    with tempfile.TemporaryDirectory(prefix='looped_moe_archive_test_') as temp:
        for r in chosen:
            target=Path(temp)/r['archive'].removesuffix('.tar.xz')
            with tarfile.open(ROOT/'archives'/r['archive'],'r:xz') as archive:
                archive.extractall(target,filter='data')
            for f in r['files']:
                assert digest(target/f['path'])==f['sha256']
                restored+=1
    checks['all_archive_sha256_checks']=len(archives)
    checks['fresh_extracted_archives']=len(chosen)
    checks['fresh_extracted_files']=restored
    write(ROOT/'final_audit.json',dict(status='PASS',checks=checks,
        scope='Integrity and reconstruction. Not additional model quality or latency evidence.'))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
