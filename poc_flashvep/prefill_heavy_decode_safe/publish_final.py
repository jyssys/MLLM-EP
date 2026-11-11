"""Publish scoped results without changing the user's checkout or index."""
import json,os,tempfile,time
from .common import *


def main():
    initialize()
    from ..completion_aware_loop_pd.publish import git,preserved,INDEX_SHA,PRIVATE_SHA
    from ..completion_aware_loop_pd.evidence_bundle import verify
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    restored=json.loads((ROOT/'state/final_burner_verification.json').read_text())
    assert restored['all_eight_verified'] and restored['all_generation_started_and_completed_round']
    bundle=verify(ROOT/'evidence_bundle')
    preserved();active=git('branch','--show-current');parent=git('rev-parse','refs/heads/'+BRANCH)
    expected='0acab1700c3b3bbbacc851d378e8b9d51d9bbdbd'
    assert parent==expected,'Inspect a newer publication before adding another commit'
    reports=REPO/'poc_flashvep/reports'
    documents=[reports/name for name in (
        'ouro_huginn_prefill_heavy_decode_safe_loop_offloading_spec.md',
        'ouro_huginn_prefill_heavy_decode_safe_measurement_search_addendum.md',
        'ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md')]
    assert all(p.is_file() for p in documents)
    names=set(documents)
    names.update(p for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.cc') and '__pycache__' not in p.parts)
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if {'worker_source','__pycache__'}&set(rel.parts) or p.suffix in ('.pyc','.lock'):continue
        if p.name in ('burner.log','final_remote_verification.json','publication_receipt.json'):continue
        # The lossless bundle retains large raw records. Small JSON stays directly readable.
        if 'raw' in rel.parts and not (p.suffix=='.json' and p.stat().st_size<=500_000):continue
        if p.stat().st_size>=90_000_000:continue
        names.add(p)
    inventory=ROOT/'state/final_publication_inventory.json'
    write(inventory,dict(time=time.time(),files=len(names)+1,parent=parent,
        direct_bytes=sum(p.stat().st_size for p in names),
        bundled_original_bytes=bundle['original_bytes'],bundled_bytes=bundle['bundled_bytes'],
        scope='Only this campaign source, its three reports, and its evidence namespace are added. Large raw records reconstruct from the verified lossless bundle. The active checkout and staged index remain unchanged.'))
    names.add(inventory);paths=sorted(str(p.relative_to(REPO)) for p in names)
    with tempfile.TemporaryDirectory(prefix='prefill_discovery_publication_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'));git('read-tree',parent,env=env)
        for i in range(0,len(paths),100):git('add','-f','--',*paths[i:i+100],env=env)
        changed=git('diff','--cached','--name-only',parent,env=env).splitlines()
        assert changed and set(changed)<=set(paths)
        code=[p for p in changed if p.endswith(('.py','.cc'))]
        if code:git('diff','--cached','--check',parent,'--',*code,env=env)
        tree=git('write-tree',env=env)
        commit=git('commit-tree',tree,'-p',parent,'-m','Measure Prefill-heavy discovery and preserve gated Loop offloading results',env=env)
        git('update-ref','refs/heads/'+BRANCH,commit,parent)
        git('push','origin',commit+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==commit
        receipt=ROOT/'state/publication_receipt.json'
        write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=commit,
            evidence_remote_SHA=remote,verified=True,files=len(paths),index_entry_SHA256=INDEX_SHA,
            private_runtime_diff_SHA256=PRIVATE_SHA,bundle_original_bytes=bundle['original_bytes'],
            bundle_bytes=bundle['bundled_bytes'],
            scope='This receipt records the verified evidence commit. A metadata child and separate remote check follow.'))
        git('add','-f','--',str(receipt.relative_to(REPO)),env=env)
        final=git('commit-tree',git('write-tree',env=env),'-p',commit,'-m','Record verified discovery publication',env=env)
        git('update-ref','refs/heads/'+BRANCH,final,commit)
        git('push','origin',final+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==final
    preserved();assert git('branch','--show-current')==active
    write(ROOT/'state/final_remote_verification.json',dict(time=time.time(),branch=BRANCH,
        commit=final,remote_SHA=remote,verified=True))
    print(json.dumps(dict(branch=BRANCH,commit=final,remote_SHA=remote,verified=True,files=len(paths))))


if __name__=='__main__':main()
