"""Index observed evidence without promoting a source read to a GPU result."""
import argparse
import datetime as dt
import json
import re
import shutil
from urllib.parse import urlsplit, urlunsplit
from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha, write

JSONL_INPUTS=['literature_sources.jsonl','parallel_sampling_sources.jsonl',
    'operator_ownership_sources.jsonl','dynamic_runtime_sources.jsonl','certified_operator_sources.jsonl']
JSON_INPUTS=['coordinator_sources.json','independent_candidate_seeds.json',
    'persistent_execution_audit.json','layout_residual_cpu.json',
    'certified_decision_cpu.json','branch_ownership_cost_ledger.json']

def canonical(record):
    text=' '.join(str(record.get(k,'')) for k in ['identifier','url','paper_url','pdf_url'])
    # A DOI suffix can resemble an arXiv number. Require a valid YYMM month.
    found=re.search(r'(?<!\d)(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?(?!\d)',text)
    if found:return 'arxiv:'+found.group(1)
    url=record.get('url') or record.get('primary_url') or record.get('code_url')
    if url:
        parts=urlsplit(url)
        return 'url:'+urlunsplit((parts.scheme,parts.netloc,parts.path.rstrip('/'),'',''))
    return 'unresolved:'+str(record.get('source_id',record.get('id',record.get('title','unknown'))))

def indexed_sources(value,pointer):
    """Extract actual source objects from heterogeneous, explicitly selected containers."""
    if isinstance(value,dict):
        if any(k in value for k in ['url','primary_url','paper_url','identifier']):
            yield pointer,value
        else:
            for key,child in value.items():yield from indexed_sources(child,pointer+'/'+str(key))
    elif isinstance(value,list):
        for index,child in enumerate(value):yield from indexed_sources(child,pointer+'/'+str(index))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    parser.add_argument('--publish-current',action='store_true')
    args=parser.parse_args();target=ROOT/'analysis'/args.label
    assert not target.exists(),target
    source_records=[];search_records=[];other_records=[];provenance=[]
    for name in JSONL_INPUTS:
        path=ROOT/'artifacts'/name
        if not path.exists():continue
        provenance.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        for number,line in enumerate(path.read_text().splitlines(),1):
            row=json.loads(line);record={'origin':str(path.relative_to(ROOT)),
                'line':number,'original':row}
            kind=row.get('record_type','source')
            if kind in ['source','source_metadata_only']:source_records.append(record)
            elif kind in ['search','search_batch','repository_search']:search_records.append(record)
            else:other_records.append(record)
    searches=ROOT/'artifacts/literature_searches.jsonl'
    if searches.exists():
        provenance.append({'path':str(searches.relative_to(ROOT)),'sha256':sha(searches)})
        for number,line in enumerate(searches.read_text().splitlines(),1):
            search_records.append({'origin':str(searches.relative_to(ROOT)),'line':number,'original':json.loads(line)})
    for name in JSON_INPUTS:
        path=ROOT/'artifacts'/name
        if not path.exists():continue
        value=read(path);provenance.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        for key in ['sources','external_sources']:
            for pointer,row in indexed_sources(value.get(key,[]),'/'+key):
                source_records.append({'origin':str(path.relative_to(ROOT)),'json_pointer':pointer,'original':row})
        for key in ['searches','search_batches']:
            for number,row in enumerate(value.get(key,[])):
                search_records.append({'origin':str(path.relative_to(ROOT)),'json_pointer':f'/{key}/{number}','original':row})
        # Some audit records keep sources beneath their own metadata object.
        meta=value.get('audit_metadata',{})
        for key in ['sources','primary_sources']:
            for pointer,row in indexed_sources(meta.get(key,[]),'/audit_metadata/'+key):
                source_records.append({'origin':str(path.relative_to(ROOT)),
                    'json_pointer':pointer,'original':row})
    groups={}
    for record in source_records:groups.setdefault(canonical(record['original']),[]).append(record)
    normalized=[]
    for key,records in sorted(groups.items()):
        normalized.append({'source_key':key,'distinct_work_count':1,
            'observations':records,'evidence_class':'SOURCE_REPORTED',
            'counting_rule':'Paper versions and associated observations share one work key. Documents with different URLs can remain distinct implementation sources.',
            'metadata_rule':'Retain original fields. Missing author, version, or read-depth fields stay unresolved.'})
    experiments=[]
    for directory in sorted((ROOT/'outputs').iterdir()):
        if not directory.is_dir():continue
        contract=directory/'run_contract.json';state=directory/'guard_status.json'
        row={'experiment_id':directory.name,'path':str(directory.relative_to(ROOT)),
             'run_contract':read(contract) if contract.exists() else None,
             'guard_status':read(state) if state.exists() else {'status':'RUNNING_OR_UNRESOLVED'},
             'files':[{ 'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
                for p in sorted(directory.rglob('*.json')) if p.name!='clean_start.json']}
        # Avoid publishing process environment inventories from the guard.
        if row['run_contract'] is not None:
            row['run_contract'].pop('environment',None)
        experiments.append(row)
    target.mkdir(parents=True)
    def jsonl(name,records):
        (target/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    jsonl('source_registry.jsonl',normalized)
    jsonl('search_log.jsonl',search_records)
    jsonl('implementation_records.jsonl',other_records)
    jsonl('experiment_manifest.jsonl',experiments)
    claims=ROOT/'artifacts/semantics_claims.jsonl'
    claim_records=[json.loads(s) for s in claims.read_text().splitlines()] if claims.exists() else []
    jsonl('claim_ledger.jsonl',claim_records)
    write(target/'index.json',{'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_observations':len(source_records),'unique_work_or_document_keys':len(normalized),
        'unique_arxiv_works':sum(r['source_key'].startswith('arxiv:') for r in normalized),
        'search_records':len(search_records),'experiment_directories':len(experiments),
        'claim_records':len(claim_records),'input_provenance':provenance,
        'scope':'Generated evidence index. This is not a claim that every source received a deep read.',
        'time_rule':'Source counts do not establish ten substantive hours. Worker time overlaps coordinator time.'})
    if args.publish_current:
        for name in ['source_registry.jsonl','search_log.jsonl','claim_ledger.jsonl','experiment_manifest.jsonl']:
            shutil.copyfile(target/name,ROOT/'artifacts'/name)
        write(ROOT/'artifacts/ledger_current.json',dict(snapshot=str(target.relative_to(ROOT)),
            index_sha256=sha(target/'index.json'),rule='Current generated index. Versioned inputs and snapshots remain unchanged.'))
    brief=read(target/'index.json')
    print(json.dumps({k:brief[k] for k in ['source_observations','unique_arxiv_works','unique_work_or_document_keys','experiment_directories','search_records']}))

if __name__=='__main__':main()
