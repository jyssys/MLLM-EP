"""Read-only progress and early parity checks for the adapter qualification."""
import json
import time
from .common import ROOT


def requests(folder):
    path=folder/'requests.jsonl';result={}
    if not path.exists():return result
    for line in path.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if not row.get('warmup'):result[row['sample_id']]=row
    return result


def main():
    state=json.loads((ROOT/'state/huginn_adapter_qualification.json').read_text())
    result=dict(qualification=state['status'],time_KST=time.strftime('%H:%M:%S'),runs=[],early_failures=[])
    for path in sorted((ROOT/'raw').glob('huginn_down_norm_adapter_qual_N[24]00*/status.json')):
        status=json.loads(path.read_text());data=requests(path.parent)
        result['runs'].append(dict(name=path.parent.name.replace('huginn_down_norm_adapter_qual_',''),
                                  status=status['status'],completed=len(data)))
        if path.parent.name.endswith('_PD_G1_0'):
            prefix=path.parent.name[:-len('_PD_G1_0')]
            for suffix in ('_PD_G1_1','_B_2'):
                other=requests(ROOT/'raw'/(prefix+suffix))
                for i in sorted(data.keys()&other.keys()):
                    a,b=data[i]['output_ids'],other[i]['output_ids']
                    if a!=b:
                        first=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                        result['early_failures'].append(dict(run=prefix+suffix,sample_id=i,first_divergence=first))
    p=ROOT/'state/huginn_adapter_postcheck.json'
    result['postcheck']=json.loads(p.read_text()).get('status') if p.exists() else 'NOT_STARTED'
    print(json.dumps(result))


if __name__=='__main__':main()
