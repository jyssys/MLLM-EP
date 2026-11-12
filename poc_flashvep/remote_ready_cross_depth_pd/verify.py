"""Verify the bounded audit and preserve every pre-existing resource owner."""
import ast,io,json,re,subprocess,time,unittest
from .common import *


def main():
    initialize();preserve()
    gate=json.loads((ROOT/'state/phase_gate.json').read_text());assert gate['equivalence_verified']
    reports=[('3b3633dbc64bcd916c2ba1ff513deb4226cc647d','ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md'),
        ('b53a8fed36a1706ac74754b53753f20edb8b913e','rcc_pd_fixed_cut_cobatching_results.md'),
        ('0efa68913f038bc2b9cab501d5c1b6347d130bbf','ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md')]
    pinned=[]
    for commit,name in reports:
        path=REPO/'poc_flashvep/reports'/name
        blob=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)
        same=hashlib.sha256(blob).hexdigest()==digest(path);assert same
        pinned.append(dict(path=str(path.relative_to(REPO)),commit=commit,SHA256=digest(path),pinned_blob_matches=same))
    write(ROOT/'references/pinned_report_verification.json',dict(reports=pinned))
    manifest=json.loads((ROOT/'state/input_evidence_manifest.json').read_text());parent_matches=[]
    for row in manifest['files']:
        path=REPO/row['path'];assert digest(path)==row['SHA256']
        blob=subprocess.check_output(['git','show',PARENT+':'+row['path']],cwd=REPO)
        same=hashlib.sha256(blob).hexdigest()==row['SHA256'];assert same
        parent_matches.append(dict(path=row['path'],SHA256=row['SHA256'],publication_parent_blob_matches=same))
    write(ROOT/'state/archived_input_verification.json',dict(files=parent_matches,
        original_evidence_preserved=True,publication_parent=PARENT,
        scope='All input bytes are unchanged and already exist in the publication parent. No duplicate raw-evidence bundle is required.'))
    suite=unittest.defaultTestLoader.loadTestsFromName('poc_flashvep.remote_ready_cross_depth_pd.test_audit')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert result.wasSuccessful(),log.getvalue()
    sources=[]
    for path in sorted(SOURCE.glob('*.py')):
        ast.parse(path.read_text(),filename=str(path));sources.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    service=owner();snapshot=resource_snapshot()
    if not snapshot['all_eight_active']:
        service.restore(list(range(8)));snapshot=resource_snapshot()
    assert snapshot['all_eight_active']
    logs={Path('/proc',str(pid),'fd/1').resolve() for pid in service.burners().values()}
    offsets={p:p.stat().st_size for p in logs};fresh={};snippets={};deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        for path,start in offsets.items():
            with path.open('rb') as stream:stream.seek(start);text=stream.read().decode(errors='replace')
            snippets[str(path)]=text
            for gpu,rounds,tokens in re.findall(r'\[GPU (\d+)\] rounds=(\d+) generated_tokens=(\d+)',text):
                fresh[int(gpu)]=dict(rounds=int(rounds),generated_tokens=int(tokens))
        if set(fresh)==set(range(8)):break
        time.sleep(2)
    assert set(fresh)==set(range(8)),'Fresh generation is missing on some approved GPU'
    write(ROOT/'state/final_resources.json',dict(resource_snapshot(),fresh_generation=fresh,
        burners_stopped_for_this_audit=False,new_GPU_experiments=0,
        scope='CPU-only audit. All approved workers remain active. Fresh completed generation confirms ongoing work.'))
    (ROOT/'state/burner_verification_excerpt.log').write_text(''.join(snippets.values()))
    missing=[]
    for name in ('ouro_huginn_remote_ready_cross_depth_pd_poc_spec.md','ouro_huginn_remote_ready_cross_depth_pd_poc_final_summary.md'):
        path=REPO/'poc_flashvep/reports'/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            resolved=path.parent/target.split('#')[0]
            if resolved==ROOT/'state/verification.json':continue
            if not resolved.exists():missing.append(dict(document=name,target=target))
    assert not missing,missing
    preserve()
    write(ROOT/'state/verification.json',dict(time=time.time(),elapsed_wall_hours=(time.time()-START)/3600,
        tests_run=result.testsRun,tests_passed=result.wasSuccessful(),test_output=log.getvalue(),source_AST_and_hashes=sources,
        pinned_reports_match=True,input_evidence_unchanged=True,existing_sources_unchanged=True,
        all_eight_approved_burners_generating=True,user_state_preserved=True,signatures=signatures(),
        new_GPU_experiments='NOT_RUN',passed=True))
    print('VERIFIED',result.testsRun,'CPU tests; all eight burners generating; original evidence and user state preserved',flush=True)


if __name__=='__main__':main()
