"""Publish the scoped final evidence through a private Git index."""
import argparse
import json
from pathlib import Path
import re
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,REPORTS,digest,preserve,write
from .delivery_requirement_inventory import IDLE_REPORTS
from .publication_checkpoint_v4 import compress
from .resource_closeout import require_final_closeout

RECEIPT='publication_commit_final.json'
RAW_RECEIPT='checkpoint_raw_records_v13.json'


def unarchived_raw(root,known):
    return sorted(path for path in (root/'raw').rglob('*') if path.is_file() and not path.is_symlink()
        and path.suffix in ['.json','.jsonl','.csv','.pickle'] and str(path.relative_to(REPO)) not in known)


def prepare_raw():
    preserve();require_final_closeout()
    folder=ROOT/'archives/public_checkpoint_v13';receipt=ROOT/'archives'/RAW_RECEIPT
    assert not folder.exists() and not receipt.exists(),'Do not overwrite an existing final raw pack'
    known={item['source'] for path in (ROOT/'archives').glob('checkpoint_raw_records_v*.json')
        for item in json.loads(path.read_text())['files']}
    paths=unarchived_raw(ROOT,known);assert paths
    folder.mkdir();records=[]
    for index,path in enumerate(paths):
        records.append(compress(path,folder))
        if index%50==0:print('FINAL_RAW_COMPRESSED',index+1,len(paths),flush=True)
    write(receipt,dict(time=time.time(),passed=True,files=records,
        scope='All remaining closed campaign structured raw, including failed runs and shared append-only logs. '
            'Lossless gzip round-trip hashes. This is byte integrity, not scientific qualification. '
            'Original image/media/model and large safetensors stay local with separate verified provenance and checksums.'))
    preserve();print('FINAL_RAW_PACK_ROUNDTRIP_VERIFIED',len(records),flush=True)


def report_names(source):
    names=set(REPORTS+IDLE_REPORTS)
    for path in source.glob('*.py'):
        names.update(re.findall(r"['\"](mllm_[A-Za-z0-9_]+\.md)['\"]",path.read_text()))
    return names


def publish(args):
    preserve();closeout=require_final_closeout()
    final=ROOT/'state'/RECEIPT
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    if final.exists():return helper.push_record(RECEIPT)
    required={
        'metric':ROOT/'analysis'/f'campaign_metric_audit_{args.metric_version}.json',
        'archive':ROOT/'analysis'/f'archive_integrity_closeout_{args.archive_version}.json',
        'figures':ROOT/'analysis'/f'figure_regeneration_closeout_{args.figure_version}.json',
        'delivery':ROOT/'analysis'/f'delivery_requirement_inventory_{args.delivery_version}.json',
        'summary':ROOT/'analysis/final_summary_evidence_v0.json',
        'requirements':ROOT/'analysis/final_requirement_audit_v0.json',
        'CPU':ROOT/'analysis'/args.cpu_audit,
    }
    data={key:json.loads(path.read_text()) for key,path in required.items()}
    assert not data['metric']['incomplete'] and data['archive']['passed'] and data['figures']['passed']
    assert all(r['available'] for r in data['delivery']['reports'])
    assert all(r['artifact_available'] for r in data['delivery']['figures'])
    assert data['summary']['decisions']['token_split_runtime']=='NO_GO_ON_TESTED_GRID'
    assert data['requirements']['delivery_checks_complete'] and not data['requirements']['all_scientific_target_gates_passed']
    assert data['CPU']['passed'] and data['CPU']['source_unchanged_during_tests']
    assert data['CPU']['source_SHA256']=={path.name:digest(path) for path in SOURCE.glob('*.py')},'Source changed after the final CPU suite'
    test=ROOT/'logs'/args.cpu_log
    match=re.search(r'Ran (\d+) tests',test.read_text());assert match and int(match[1])>=245
    assert test.read_text().rstrip().endswith('OK')
    paths=list(SOURCE.glob('*.py'))+[test,ROOT/'resource_ledger.jsonl']+list(required.values())
    missing=[]
    for name in sorted(report_names(SOURCE)):
        path=ROOT.parent/name
        if path.is_file():paths.append(path)
        elif name in REPORTS+IDLE_REPORTS:missing.append(name)
    assert not missing,missing
    for directory in ['analysis','configs','state','data','references','figures','logs']:
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            if directory=='references' and path.suffix not in ['.json','.csv','.zip','.py','.cpp','.cu','.h','.sh']:continue
            if directory=='logs' and path.suffix not in ['.log','.json']:continue
            if directory in ['analysis','configs','state','data'] and path.suffix not in ['.json','.jsonl','.csv']:continue
            if directory=='figures' and path.suffix not in ['.csv','.png','.pdf']:continue
            paths.append(path)
    for pattern in ['*.source.zip','checkpoint_raw_records_v*.json']:
        paths.extend((ROOT/'archives').glob(pattern))
    for receipt in (ROOT/'archives').glob('checkpoint_raw_records_v*.json'):
        for record in json.loads(receipt.read_text())['files']:
            paths.append(REPO/record['compressed_path'])
    # Original local tensor bytes are not copied into Git. The final integrity
    # audit contains their byte sizes, hashes and earlier checksum references.
    local=[];selected=[]
    for path in sorted(set(paths)):
        assert path.is_file() and REPO in path.resolve().parents
        if path.stat().st_size>=90*1024**2:
            local.append(dict(path=str(path.relative_to(REPO)),bytes=path.stat().st_size,SHA256=digest(path),
                scope='Large derived artifact retained locally. No raw deletion or lossy replacement.'))
        else:selected.append(path)
    readme=ROOT/'README_final.md';assert not readme.exists()
    readme.write_text('\n'.join([
        '# MLLM EPD final campaign handoff','',
        'The requested experiment campaign is closed. Scientific result: HOLD and NO_GO_ON_TESTED_GRID.',
        'This is not a claim that model quality, prefill-heavy, safe headroom or online token-split gain passed.',
        'Original Main400 is unchanged. Later experiments use the user-approved frozen Main100.',
        'Actual online encoding, continuous baselines, exact multimodal token continuation and Natural/replay N3 evidence are preserved.',
        'Static10 passed its scoped decode gates but failed the full serving improvement contract.',
        'Qualified low-load physical E/D idle is not high-load actionable slack.',
        'Probe protection and high-load observer failures remain failures. No active-decode unsafe insertion was used.',
        'C46 stopped on a V2 observer API error. Its four completed methods were reused. Only the two unfinished baselines ran in C46R1.',
        'Raw timestamp metrics, source archives, original tensor checksums and required figures were checked separately.',
        'Four historical append-only raw snapshots and three legacy boot metadata discrepancies are disclosed, not overwritten. Immutable boot ZIP/materialization and original startup ledger provide separate source evidence.',
        'Figure reproduction compares CSV and PNG pixels. PDF header/EOF is not full PDF-render verification.',
        'Original media/model/tensor bytes stay local. Git contains scoped source, structured raw, lossless gzip, tables, figures and checksum receipts.',
        'All approved GPU0-7 burners were restored and ownership and utilization were verified.',
        'User checkout, staged index and old Dense AR evidence were preserved.',
        'The Korean final summary is ../mllm_epd_prefill_heavy_token_split_final_summary.md.',
        'The local publication receipt records the actual remote SHA after this commit is pushed.',
        'No future commit hash is guessed or embedded in its own tree.','']))
    selected.append(readme)
    selection=ROOT/'state/final_publication_selection_v0.json';assert not selection.exists()
    write(selection,dict(time=time.time(),files=len(selected),bytes=sum(p.stat().st_size for p in selected),
        local_only_large_derived=local,original_tensors='Verified in archive_integrity_closeout; retained locally',
        requirements={key:dict(path=str(path.relative_to(REPO)),SHA256=digest(path)) for key,path in required.items()},
        final_resource_closeout_SHA256=digest(ROOT/'state/final_resource_closeout_v0.json'),
        closed_GPU_leases=closeout['closed_GPU_leases'],CPU_tests=int(match[1]),
        scientific_decisions=data['summary']['decisions'],
        scope='Final scoped publication selection. Remote verification follows actual normal push. No checkout or staged-index mutation.'))
    selected.append(selection)
    result=helper.commit_paths(selected,'MLLM EPD: close Main100 online token-split and E/D idle feasibility campaign with qualified failures',RECEIPT)
    assert result['passed'] and result['remote_verified'];preserve()
    print('FINAL_CAMPAIGN_REMOTE_SHA_VERIFIED',result['remote_SHA'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-raw',action='store_true')
    parser.add_argument('--metric-version',default='v4');parser.add_argument('--archive-version',default='v0')
    parser.add_argument('--figure-version',default='v0');parser.add_argument('--delivery-version',default='v4')
    parser.add_argument('--cpu-log',default='cpu_regression_final_v0.log')
    parser.add_argument('--cpu-audit',default='final_cpu_regression_v0.json');args=parser.parse_args()
    prepare_raw() if args.prepare_raw else publish(args)


if __name__=='__main__':main()
