"""Score unchanged outputs with pinned task code. Never fabricate extraction."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import types
import time
from typing import List

import numpy as np
import regex

from .common import ROOT,digest,preserve,write


def source_definitions(filename,names,namespace):
    path=ROOT/'references/quality_sources'/filename
    receipt=json.loads(path.read_text());text=receipt['source_text']
    nodes=[node for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return dict(receipt=str(path),receipt_SHA256=digest(path),source_text_SHA256=hashlib.sha256(text.encode()).hexdigest(),
                selected_definitions=sorted(names),upstream_url=receipt.get('url'))


def scorers():
    ns={}
    ml_functions=['levenshtein_distance','anls_compute','is_float_equal','get_clean_string','is_exact_match','isfloat','eval_score']
    from math import isclose
    ml_ns=dict(re=regex,isclose=isclose,eval=ast.literal_eval)
    ml=source_definitions('MMLongBench-Doc_eval_eval_score.py.json',ml_functions,ml_ns)
    ns.update(ml_ns)
    doc_ns=dict(editdistance=types.SimpleNamespace(eval=ns['levenshtein_distance']))
    doc=source_definitions('MPDocVQA_metrics.py.json',['Evaluator'],doc_ns)
    mantis_ns=dict(re=regex,List=List)
    mantis=source_definitions('Mantis_mantis_benchmark_eval.py.json',['parse_answer','get_option','get_prediction'],mantis_ns)
    return doc_ns['Evaluator'](),mantis_ns['get_prediction'],ml_ns['eval_score'],dict(DocVQA=doc,Mantis=mantis,MMLongDoc=ml)


def score_request(row,prediction,scoring):
    doc,mantis,ml,_=scoring
    source=row['source'];references=row['reference_answers']
    if source in ['lmms-lab/DocVQA','lmms-lab/MP-DocVQA']:
        value=doc.get_metrics([list(map(str,references))],[prediction])
        return dict(score=float(value['anls'][0]),accuracy=float(value['accuracy'][0]),
                    metric='Organizer-framework ANLS on unchanged raw response',complete_task_scorer=True)
    if source=='TIGER-Lab/Mantis-Eval':
        value=mantis(row['metadata']['question_type'],prediction,str(references[0]),row['options'],'Mantis-Eval')
        return dict(score=float(value['correct']),parsed_answer=value['parsed_answer'],
                    metric='Pinned Mantis get_prediction',complete_task_scorer=True)
    if source=='yubo2333/MMLongBench-Doc':
        try:value=ml(str(references[0]),prediction,row['metadata']['answer_format'])
        except (ValueError,TypeError,SyntaxError,IndexError):value=None
        return dict(score=value,metric='Official scoring function on RAW answer, without required GPT extraction',
                    complete_task_scorer=False,official_full_pipeline='NOT_RUN',reason='No authorized external API/key. No answer was silently extracted.')
    raise ValueError(source)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-run',required=True);p.add_argument('--runs',required=True);p.add_argument('--name',required=True)
    args=p.parse_args();preserve()
    manifest=json.loads(args.manifest.read_text());rows=manifest['requests'];lookup={r['request_id']:r for r in rows}
    scoring=scorers();results={};ref={}
    names=[args.reference_run]+args.runs.split(',')
    for name in names:
        path=ROOT/'raw'/f'{name}_requests.jsonl'
        observations=[json.loads(s) for s in path.read_text().splitlines()]
        selected={r['request_id']:r for r in observations if r['request_id'] in lookup}
        assert len(selected)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in selected.values()),name
        details=[];groups=defaultdict(list)
        for row in rows:
            item=selected[row['request_id']]
            assert item['prompt_SHA256']==row['prompt_SHA256']
            value=score_request(row,item['text'],scoring)
            detail=dict(request_id=row['request_id'],source=row['source'],raw_answer=item['text'],
                        reference_answers=row['reference_answers'],output_tokens=item['output_tokens'],**value)
            details.append(detail);groups[row['source']].append(detail)
        by_source={source:dict(requests=len(items),scored=sum(r['score'] is not None for r in items),
            mean_score=float(np.mean([r['score'] for r in items if r['score'] is not None])),
            complete_task_scorer=all(r['complete_task_scorer'] for r in items),metric=items[0]['metric']) for source,items in groups.items()}
        if name==args.reference_run:ref=by_source
        else:
            for source,v in by_source.items():
                v['difference_vs_reference_pp']=100*(v['mean_score']-ref[source]['mean_score'])
                v['point_estimate_within_2pp_margin']=v['difference_vs_reference_pp']>=-2
        results[name]=dict(raw_SHA256=digest(path),requests=details,source_results=by_source,
                           overall_gate='PENDING: MMLongDoc official answer extraction not available')
    write(ROOT/'analysis'/f'{args.name}.json',dict(time=time.time(),manifest_SHA256=digest(args.manifest),
        scorer_provenance=scoring[3],runs=results,evidence='TASK_SCORE_ANALYSIS',
        scope='Source/task score definitions retained. No prompt edits or answer rewrites. '
            'The MMLongDoc RAW-score diagnostic is not its official full quality metric. '
            'Python edit distance substitutes the missing native dependency with the same Levenshtein distance. '
            'ast.literal_eval replaces unsafe eval for list literals. No arbitrary model text is executed. '
            'Point score margins are not confidence intervals or multi-start quality confirmation.'))
    print(json.dumps({name:r['source_results'] for name,r in results.items()},indent=2));preserve()


if __name__=='__main__':main()
