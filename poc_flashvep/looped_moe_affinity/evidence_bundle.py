"""Preserve raw evidence locally. Publish compact numeric evidence and hashes."""
import gzip
import json
import shutil
from .common import ROOT,digest,write

def main():
    inventory=[]
    for path in sorted((ROOT/'raw').rglob('*')):
        if path.is_file():inventory.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    write(ROOT/'raw_evidence_manifest.json',dict(files=inventory,
        raw_storage='Complete raw files remain local under raw/. They are not deleted.',
        publication='Compact numeric evidence is archived separately. Full benchmark prompts are not republished.',
        raw_bytes=sum(v['bytes'] for v in inventory)))
    traces=[]
    target=ROOT/'router_trace_sample.jsonl.gz'
    with gzip.open(target,'wt') as output:
        for model in ['ifm_ouro','ifm_huginn','foil1']:
            source=ROOT/'raw'/f'{model}_route_corpus_v2/router_corpus.jsonl.gz'
            kept=set();count=0
            with gzip.open(source,'rt') as stream:
                for line in stream:
                    row=json.loads(line)
                    key=(row['split'],row['batch'],row['context'],row['domain'],row['site'],row['recurrent'],row['loop'])
                    if key not in kept:
                        output.write(json.dumps(row)+'\n');kept.add(key);count+=1
            traces.append(dict(model=model,path=source.relative_to(ROOT).as_posix(),sha256=digest(source),
                bytes=source.stat().st_size,published_sample_rows=count,sampled_positions_per_request=64,
                calibration_prompts_per_domain=8,heldout_prompts_per_domain=16,
                split_unit='Request/prompt, not tokens from the same prompt',
                qualification='Uniform token-position samples, not complete per-token traces. Batch shapes use repeated or truncated public text.'))
    write(ROOT/'router_trace_manifest.json',dict(corpora=traces,safe_sample=target.name,
        sample_selection='First record per shape, domain, physical site, pass and split. Audit sample, not the full statistical corpus.',
        complete_route_storage='raw/{model}_route_corpus_v2/router_corpus.jsonl.gz',
        full_counts='New native hardware controls save every router ID separately. See full_count_prediction.csv.'))
    archive=ROOT/'archives';archive.mkdir(exist_ok=True)
    ep_routes=ROOT/'ep_affinity_metrics.csv'
    with ep_routes.open('rb') as source,gzip.open(ROOT/'ep_affinity_metrics.csv.gz','wb') as output:
        shutil.copyfileobj(source,output)
    write(ROOT/'quality/compressed_table_provenance.json',dict(source=ep_routes.name,sha256=digest(ep_routes),
        compressed='ep_affinity_metrics.csv.gz',
        scope='The complete CSV remains local. The tracked gzip contains the same bytes. No rows are removed.'))
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            source=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
            with gzip.open(archive/f'{model}_ep{ep}_numeric_baseline.jsonl.gz','wt') as output:
                for path in sorted(source.iterdir()):
                    if path.suffix not in ['.json','.jsonl'] or path.name=='routes.jsonl':continue
                    if path.suffix=='.jsonl':data=[json.loads(v) for v in path.read_text().splitlines()]
                    else:data=json.loads(path.read_text())
                    output.write(json.dumps(dict(source=path.relative_to(ROOT).as_posix(),data=data))+'\n')
            kernel=source/'kineto_b1_p128.rank0.json.gz'
            if kernel.exists():shutil.copyfile(kernel,archive/f'{model}_ep{ep}_representative_rank0_cupti.json.gz')
            generic=ROOT/'raw'/f'{model}_ep{ep}_generic_metadata_controls_v1'
            with gzip.open(archive/f'{model}_ep{ep}_generic_metadata.jsonl.gz','wt') as output:
                for path in sorted(generic.iterdir()):
                    if path.suffix not in ['.json','.jsonl']:continue
                    data=([json.loads(v) for v in path.read_text().splitlines()] if path.suffix=='.jsonl' else json.loads(path.read_text()))
                    output.write(json.dumps(dict(source=path.relative_to(ROOT).as_posix(),data=data))+'\n')
    failures=[]
    for path in sorted((ROOT/'raw').rglob('failure.rank*.json')):
        failures.append(dict(source=path.relative_to(ROOT).as_posix(),data=json.loads(path.read_text())))
    write(ROOT/'archives/failed_attempts.json',dict(failures=failures,
        scope='Original saved failure tracebacks. A preserved failure is not a successful experiment.'))
    sizes=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(archive.rglob('*')) if p.is_file() and p.name!='manifest.json']
    write(ROOT/'archives/manifest.json',dict(files=sizes,
        scope='Numeric baseline and generic controls, failure records, and representative rank-0 CUPTI observers. All-rank raw traces remain local.',
        evidence='Actual archived records, not regenerated timing values.'))

if __name__=='__main__':main()
