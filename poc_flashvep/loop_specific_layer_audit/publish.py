"""Verify the CPU audit, preserve user state, and publish a small child branch."""
import contextlib,io,json,os,subprocess,tempfile,time
from pathlib import Path
from .audit import REPO,ROOT,sha,write

BRANCH='flashvep/loop-specific-vs-dense-layer-20261112'
PARENT='0efa68913f038bc2b9cab501d5c1b6347d130bbf'
SOURCE=Path(__file__).parent
REPORT=REPO/'poc_flashvep/reports/ouro_huginn_loop_specific_vs_dense_ar_layer_split_analysis.md'


def main():
    from ..completion_aware_loop_pd.publish import git,preserved,INDEX_SHA,PRIVATE_SHA
    from ..prefill_heavy_decode_safe.common import configure
    preserved();active=git('branch','--show-current')
    assert git('ls-remote','origin','refs/heads/flashvep/prefill-heavy-decode-safe-20261111').split()[0]==PARENT
    value=json.loads((ROOT/'audit.json').read_text());assert value['CPU_checks']['passed']
    text=REPORT.read_text()
    for item in value['models']:
        assert str(item['unique_parameters']) in text
        assert str(item['parameter_groups']['shared_recurrent']) in text
    for item in value['attention_accounting']:
        assert str(item['loop_causal_query_key_pairs_per_core_attention']) in text
        assert str(item['token_causal_query_key_pairs_per_core_attention']) in text
    checks=[]
    for cmd in (
        ['python','-m','compileall','-q',str(SOURCE.relative_to(REPO))],
        ['python','/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',str(REPORT),
            '--disable','synonym-rotation','--json']):
        r=subprocess.run(cmd,cwd=REPO,text=True,capture_output=True)
        checks.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0
    write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),GPU_work='NOT_RUN',
        reason='CPU architectural and prior-evidence audit complete. No new Scheduler or GPU campaign.'))
    from ..completion_aware_loop_pd import restore_burners
    restore_burners.ROOT=ROOT;restore_burners.configure=configure
    restore_burners.main()
    write(ROOT/'state/final_checks.json',dict(time=time.time(),passed=True,checks=checks,
        report_SHA256=sha(REPORT),source_SHA256={str(p.relative_to(REPO)):sha(p) for p in SOURCE.glob('*.py')},
        original_index_SHA256=INDEX_SHA,external_runtime_diff_SHA256=PRIVATE_SHA,
        scope='CPU algebra, metadata consistency, syntax, report clarity, unchanged user state, and actual generation on all eight GPUs. No new model/GPU qualification is implied.'))
    paths=sorted([str(REPORT.relative_to(REPO)),
        *[str(p.relative_to(REPO)) for p in SOURCE.glob('*.py')],
        *[str(p.relative_to(REPO)) for p in ROOT.rglob('*') if p.is_file() and p.suffix=='.json']])
    assert subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+BRANCH],cwd=REPO).returncode!=0
    with tempfile.TemporaryDirectory(prefix='loop_layer_audit_publication_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'));git('read-tree',PARENT,env=env)
        git('add','-f','--',*paths,env=env)
        changed=git('diff','--cached','--name-only',PARENT,env=env).splitlines();assert changed and set(changed)<=set(paths)
        git('diff','--cached','--check',PARENT,env=env)
        commit=git('commit-tree',git('write-tree',env=env),'-p',PARENT,'-m',
            'Audit Loop-specific mechanisms against equivalent logical Layer splitting',env=env)
        git('update-ref','refs/heads/'+BRANCH,commit,'0'*40)
        git('push','origin',commit+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==commit
        receipt=ROOT/'state/publication_receipt.json'
        write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=commit,remote_SHA=remote,verified=True,
            parent_discovery_commit=PARENT,files=len(paths),scope='Read-only source and existing-evidence audit. No new online benefit claim.'))
        git('add','-f','--',str(receipt.relative_to(REPO)),env=env)
        final=git('commit-tree',git('write-tree',env=env),'-p',commit,'-m','Record verified Loop versus Layer audit publication',env=env)
        git('update-ref','refs/heads/'+BRANCH,final,commit);git('push','origin',final+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==final
    preserved();assert git('branch','--show-current')==active
    write(ROOT/'state/final_remote_verification.json',dict(time=time.time(),branch=BRANCH,commit=final,remote_SHA=remote,verified=True))
    print(json.dumps(dict(branch=BRANCH,commit=final,remote_SHA=remote,verified=True)))


if __name__=='__main__':main()
