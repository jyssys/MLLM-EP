"""Audit existing Stock compiler artifacts without executing cached code."""
import hashlib
import json
from pathlib import Path
import re
import time
from .common import ROOT,write,digest
from .coordinator_state_fix import compare


def analyze():
    from ..dense_ar_output_aware_v2.cache_artifact_audit import inspect
    names=['PC_CoordQual_StockToken_NormalGC_Strict8','PC_CoordQual_StockToken_DeferredGC_Strict8']
    roles=[]
    for role in ('prefill','decode'):
        roots=[]
        for name in names:
            text=(ROOT/'raw'/name/(role+'.log')).read_text()
            match=re.search(r'Using cache directory: (.*?) for vLLM',text);assert match
            roots.append(Path(match[1]))
        paths=[{path.name:path for path in root.glob('artifact_compile_range_*_subgraph_*')} for root in roots]
        assert paths[0].keys()==paths[1].keys()
        results=[]
        for filename in sorted(paths[0]):
            records=[inspect(value[filename]) for value in paths]
            maps=[{(row['kind'],row['key']):row for row in record['rows']} for record in records]
            shared=maps[0].keys()&maps[1].keys();differences=[]
            for key in sorted(shared):
                a,b=[value[key] for value in maps]
                normalize=lambda row:{k:v for k,v in (row['autotune_configuration'] or {}).items() if k!='time_taken_ms'}
                codes=lambda row:[item['kernel_path_comments_removed_SHA256'] for item in row['generated_code_string_hashes']]
                if normalize(a)!=normalize(b) or codes(a)!=codes(b):
                    differences.append(dict(kind=key[0],key=key[1],autotune=[a['autotune_configuration'],b['autotune_configuration']],
                        generated_code_equal_without_path_comments=codes(a)==codes(b)))
            results.append(dict(artifact=filename,cache_key_equal=records[0]['cache_key']==records[1]['cache_key'],
                artifact_key_sets_equal=maps[0].keys()==maps[1].keys(),semantic_differences=differences,
                raw_SHA256=[digest(value[filename]) for value in paths]))
        roles.append(dict(role=role,cache_roots=list(map(str,roots)),artifacts=results))
    state=compare(*names)
    source=Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/compilation/backends.py')
    value=dict(time=time.time(),roles=roles,strict_state=state,installed_backends_SHA256=digest(source),
        evidence='TRACE_ANALYSIS',scope='Read checksummed artifact containers and pickle opcodes only. No unpickling or GPU context. '
            'The installed compiler includes traced absolute source paths in its code hash. Raw jobs use distinct worker_source paths. '
            'Different outer cache roots can trigger independent compilation. They do not alone prove a numerical cause. '
            'Equal extracted code and autotune settings do not exclude external library kernels,attention metadata or runtime ordering. '
            'No strict failure is waived and no old artifact is edited.')
    write(ROOT/'analysis/coordinator_stock_compiler_artifact_audit.json',value)
    print('STOCK_ARTIFACT_AUDIT',[(r['role'],len(r['artifacts']),sum(bool(a['semantic_differences']) for a in r['artifacts'])) for r in roles],flush=True)
    return value


if __name__=='__main__':analyze()
