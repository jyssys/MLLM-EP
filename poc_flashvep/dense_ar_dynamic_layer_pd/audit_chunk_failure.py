"""Audit real request identity and query coverage before blaming kernel math."""
from .common import *
from .analyze import records


def main():
    folder=ROOT/'raw/coarse_O16_N64_q4.777_chunk2048_s1_v1'
    checks=[]
    for role in ('replica0','replica1'):
        identity={r['request_id']:r['external_request_id'] for path in folder.glob(f'extension_{role}_*.jsonl')
            for r in records(path) if r['event']=='REQUEST_IDENTITY'}
        trace=json.loads(next(folder.glob(f'native_{role}_*.json')).read_text())
        by_request={}
        for row in trace['prompt_spans']:
            rid=identity.get(row['request_id'],'')
            if not (rid.startswith('q') and rid[1:].isdigit()):continue
            by_request.setdefault(rid,[]).append(row)
        for rid,spans in sorted(by_request.items()):
            ordered=sorted(spans,key=lambda r:r['start']);cursor=0;passed=True
            for row in ordered:
                passed=passed and row['start']==cursor and row['count']>0
                cursor=row['start']+row['count']
            checks.append(dict(request=rid,role=role,queries=cursor,chunks=len(spans),
                query_ranges=[[r['start'],r['start']+r['count']] for r in ordered],
                no_duplicate_or_missing_query=passed and cursor==8000))
    assert len(checks)==64
    result=dict(request_query_coverage=checks,all_query_coverage_pass=all(r['no_duplicate_or_missing_query'] for r in checks),
        evidence='TRACE_ANALYSIS',kernel_numerical_root_cause='NOT_IDENTIFIED',
        scope='Recorded Native scheduler query ranges and internal-to-external identities. This does not check every KV tensor or establish a specific numerical kernel cause.')
    write(ROOT/'analysis/streaming_chunk_query_coverage.json',result)
    print('QUERY_COVERAGE',len(checks),result['all_query_coverage_pass'])


if __name__=='__main__':main()
