"""Preserve full raw bytes locally and publish a verified smaller evidence stream."""
import argparse
import json
import os
from pathlib import Path
import time
from .common import ROOT,REPO,write,digest,bind,preserve

SENSITIVE_KEYS={'input_ids','prompt_token_ids','original_context','original_input','reference_answers',
    'rendered_prompt_full','prompt_text','messages','context','answers'}


def sensitive(value):
    if isinstance(value,dict):
        return bool(SENSITIVE_KEYS.intersection(value)) or any(sensitive(item) for item in value.values())
    if isinstance(value,list):return any(sensitive(item) for item in value)
    return False


def public_reason(path):
    relative=path.relative_to(ROOT)
    if relative.parts[0]=='data':return 'THIRD_PARTY_CONTEXT_AND_TOKEN_DATA_LOCAL_ONLY'
    if relative.parts[0]=='references' and path.suffix=='.pdf':return 'EXTERNAL_PAPER_LOCAL_ONLY'
    if path.suffix in ('.pt','.pth','.npy','.npz','.safetensors','.so'):return 'TENSOR_OR_BUILD_OUTPUT_LOCAL_ONLY'
    if any(part in relative.parts for part in ('hidden_transport','state_transport')):return 'EXPIRED_TRANSPORT_STATE_LOCAL_ONLY'
    if path.name=='source.zip' and 'raw' in relative.parts:return 'CANONICAL_SOURCE_ZIP_PUBLISHED_ONCE'
    if path.suffix=='.log':return 'FULL_WORKER_LOG_LOCAL_ONLY_WITH_HASH'
    if path.suffix in ('.json','.jsonl'):
        try:
            values=map(json.loads,path.read_text().splitlines()) if path.suffix=='.jsonl' else [json.loads(path.read_text())]
            if any(sensitive(value) for value in values):return 'PROMPT_OR_REFERENCE_CONTENT_LOCAL_ONLY'
        except (ValueError,UnicodeDecodeError):return 'UNPARSED_RAW_CONTENT_LOCAL_ONLY'
    return None


def select():
    paths=[];omitted=[]
    for top in ('raw','data','references','logs','attention_poc'):
        for folder,dirs,names in os.walk(ROOT/top):
            for name in list(dirs):
                if name in ('worker_source','runtime_sources','__pycache__','build','.cache'):
                    omitted.append(dict(path=str((Path(folder)/name).relative_to(REPO)),reason='DUPLICATE_SOURCE_OR_REGENERABLE_BUILD'))
                    dirs.remove(name)
            for name in names:
                path=Path(folder)/name;assert not path.is_symlink(),path
                if path.suffix in ('.lock','.local') or '.tmp.' in name:
                    omitted.append(dict(path=str(path.relative_to(REPO)),reason='LOCAL_OPERATIONAL_FILE'));continue
                paths.append(path)
    return sorted(paths),omitted


def archive():
    bind();preserve()
    assert json.loads((ROOT/'state/CAMPAIGN_GPU_WORK_CLOSED.json').read_text())['passed']
    from ..dense_ar_longbench400_pd.evidence_archive import pack,split
    target=ROOT/'archives';target.mkdir(exist_ok=True)
    manifest=target/'evidence_archive_manifest.json';assert not manifest.exists()
    paths,omitted=select();local=target/'local_full_evidence.tar.zst'
    local_rows=pack(paths,local)
    public_paths=[];excluded=[]
    for path in paths:
        reason=public_reason(path)
        if reason:excluded.append(dict(path=str(path.relative_to(REPO)),reason=reason))
        else:public_paths.append(path)
    public=target/'public_essential_evidence.tar.zst';public_rows=pack(public_paths,public)
    parts=split(public,45*1024**2)
    canonical={digest(path):str(path.relative_to(REPO)) for path in (ROOT/'references/source_capsules').glob('*.zip')}
    sources=[]
    for path in paths:
        if path.name=='source.zip' and path.parent.parent==ROOT/'raw':
            sha=digest(path);assert sha in canonical
            sources.append(dict(path=str(path.relative_to(REPO)),SHA256=sha,canonical=canonical[sha]))
    value=dict(time=time.time(),passed=True,verified_lossless=True,local_archive=dict(path=str(local.relative_to(REPO)),
        bytes=local.stat().st_size,SHA256=digest(local),public=False),
        public_archive=dict(path=str(public.relative_to(REPO)),bytes=public.stat().st_size,SHA256=digest(public),parts=parts),
        local_members=local_rows,public_members=public_rows,local_only=excluded,
        duplicate_or_operational_directories=omitted,duplicate_source_references=sources,
        scope='Every decompressed member passed byte-count and SHA256 validation. Original raw files remain unchanged. '
            'All selected third-party contexts,tensors and full worker logs remain in the local lossless archive. '
            'Public evidence excludes original prompt/reference text and prompt token sequences. Generated output token IDs remain in request records. '
            'Publish canonical worker archives once. Reconstruct duplicate source trees from their exact source SHA. '
            'Input reconstruction uses the pinned official archive,templates,tokenizer and published identity rosters. '
            'Concatenate public parts in manifest order,check the merged SHA256,and decompress with zstd.')
    write(manifest,value);preserve()
    print('LOSSLESS_EVIDENCE_ARCHIVE_PASS',len(local_rows),len(public_rows),len(parts),flush=True)
    return value


def selftest():
    import tempfile
    bind()
    from ..dense_ar_longbench400_pd.evidence_archive import pack
    assert sensitive(dict(a=[dict(input_ids=[1,2])]))
    assert not sensitive(dict(input_ids_SHA256='hash',output_ids=[1,2]))
    with tempfile.TemporaryDirectory(prefix='p_critical_archive_test_') as folder:
        rows=pack([Path(__file__),ROOT/'configs/research_contract.json'],Path(folder)/'test.tar.zst')
        assert len(rows)==2
    print('LOSSLESS_EVIDENCE_SELFTEST_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    selftest() if args.selftest else archive()
