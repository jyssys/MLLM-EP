"""Inspect local compiler cache records without loading executable artifacts."""
import base64
import hashlib
import json
from pathlib import Path
import pickletools
import re
import time
import zlib
from .common import ROOT,write

class Reader:
    def __init__(self,data):
        assert zlib.crc32(data[4:]).to_bytes(4,'big')==data[:4]
        self.data=data;self.index=4
    def uint(self):
        value=int.from_bytes(self.data[self.index:self.index+8],'big');self.index+=8;return value
    def bytes(self):
        size=self.uint();value=self.data[self.index:self.index+size];self.index+=size;assert len(value)==size;return value
    def text(self):return base64.b64decode(self.bytes()).decode()
    def done(self):return self.index==len(self.data)

def inspect(path):
    reader=Reader(path.read_bytes());assert reader.bytes()==b'CacheCompiledArtifact'
    torch_key=reader.bytes();key=reader.text();inner=Reader(reader.bytes());assert reader.done() and inner.uint()==1
    rows=[]
    while not inner.done():
        kind=inner.text();count=inner.uint()
        for _ in range(count):
            artifact_key=inner.text();content=inner.bytes();strings=[];error=None
            autotune=None
            if kind=='autotune':
                try:autotune=json.loads(content)
                except (ValueError,UnicodeDecodeError):pass
            # pickletools reads opcodes. It does not unpickle or execute code.
            try:
                for op,arg,position in pickletools.genops(content):
                    if isinstance(arg,str) and ('async_compile' in arg or 'def call(args)' in arg or '@triton.jit' in arg):
                        normalized=re.sub(r'^# kernel path: .*$','',arg,flags=re.MULTILINE)
                        strings.append(dict(bytes=len(arg.encode()),SHA256=hashlib.sha256(arg.encode()).hexdigest(),
                            kernel_path_comments_removed_SHA256=hashlib.sha256(normalized.encode()).hexdigest()))
            except Exception as value:error=type(value).__name__
            rows.append(dict(kind=kind,key=artifact_key,bytes=len(content),SHA256=hashlib.sha256(content).hexdigest(),
                generated_code_string_hashes=strings,autotune_configuration=autotune,non_pickle_or_parse_status=error))
    return dict(path=str(path),torch_key_SHA256=hashlib.sha256(torch_key).hexdigest(),cache_key=key,rows=rows)

def run():
    roots=[Path('/home/esjung/.cache/vllm/torch_compile_cache')/r/'rank_0_0/backbone' for r in ('9bcbca3da3','211c2d039b')]
    comparisons=[]
    for subgraph in (0,1,36):
        records=[inspect(root/f'artifact_compile_range_1_8192_subgraph_{subgraph}') for root in roots]
        maps=[{(r['kind'],r['key']):r for r in record['rows']} for record in records]
        shared=maps[0].keys()&maps[1].keys()
        comparisons.append(dict(subgraph=subgraph,records=records,same_AOT_cache_key=records[0]['cache_key']==records[1]['cache_key'],
            artifact_key_sets_equal=maps[0].keys()==maps[1].keys(),
            shared_artifact_content_equal=[dict(kind=k[0],key=k[1],content_equal=maps[0][k]['SHA256']==maps[1][k]['SHA256'],
                generated_code_strings_equal=maps[0][k]['generated_code_string_hashes']==maps[1][k]['generated_code_string_hashes'],
                generated_code_equal_without_path_comments=
                    [s['kernel_path_comments_removed_SHA256'] for s in maps[0][k]['generated_code_string_hashes']]==
                    [s['kernel_path_comments_removed_SHA256'] for s in maps[1][k]['generated_code_string_hashes']],
                autotune_launch_choices_equal=
                    {a:b for a,b in (maps[0][k]['autotune_configuration'] or {}).items() if a!='time_taken_ms'}==
                    {a:b for a,b in (maps[1][k]['autotune_configuration'] or {}).items() if a!='time_taken_ms'}) for k in sorted(shared)]))
    result=dict(time=time.time(),comparisons=comparisons,evidence='READ_ONLY_COMPILER_ARTIFACT_AUDIT',
        scope='Parse the installed checksummed serialization schema and pickle opcodes only. Do not load code or create a CUDA context. '
            'Opaque metadata differences do not prove arithmetic differences. Equal extracted code strings do not prove equal external library kernels or runtime ordering.')
    write(ROOT/'analysis/stock_compiler_nested_artifact_audit_v13.json',result)
    print([(r['subgraph'],r['artifact_key_sets_equal'],r['shared_artifact_content_equal']) for r in comparisons],flush=True)

if __name__=='__main__':run()
