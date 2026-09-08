"""Never replace official scoring: separate obvious format from content losses."""
import argparse,json,re
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
d=pd.read_csv(a.input);base=d[d.policy=='vanilla'][['request_id','correct','prediction']]
j=d.merge(base,on='request_id',suffixes=('','_baseline'))
num=re.compile(r'(?<![\w.])[+-]?(?:\d+(?:,\d{3})*)(?:\.\d+)?%?')
rows=[]
for (dataset,policy),g in j.groupby(['dataset','policy']):
    bad=g[(g.correct_baseline==1)&(g.correct==0)]
    count=0;examples=[]
    for _,row in bad.iterrows():
        pred=str(row.prediction).strip();b=str(row.prediction_baseline).strip()
        # This only checks whether the vanilla text/value is still present; it
        # is not a semantic correctness oracle and may match a spurious number.
        present=False
        if dataset=='chartqa' and num.fullmatch(b):
            try:
                bn=float(b.rstrip('%').replace(',',''))
                values=[float(x.rstrip('%').replace(',','')) for x in num.findall(pred)]
                present=any(abs(x-bn)<=.05*abs(bn) if bn!=0 else x==0 for x in values)
            except ValueError:pass
        elif b:
            present=re.search(r'(?<!\w)'+re.escape(b.lower())+r'(?!\w)',pred.lower()) is not None
        count+=present
        if present:examples.append(dict(request=row.request_id,baseline=b,prediction=pred))
    rows.append(dict(dataset=dataset,policy=policy,official_lost_correct=len(bad),
        baseline_value_or_text_present_anywhere=count,examples=examples,
        scope='PERMISSIVE_FORMAT_DIAGNOSTIC_ONLY; no change to official scores; presence does not prove correctness'))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
