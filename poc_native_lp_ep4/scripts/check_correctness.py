import argparse
import json
from pathlib import Path

import torch

from poc_native_lp_ep4.contract import atomic_json,read_json,partition


def errors(a,b):
    a,b=a.float(),b.float()
    d=(a-b).abs()
    return {"max_abs":float(d.max()),"max_relative":float((d/a.abs().clamp_min(1e-5)).max()),
        "relative_l2":float((a-b).norm()/a.norm()),
        "min_row_cosine":float(torch.nn.functional.cosine_similarity(a,b,dim=-1).min())}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    waves=read_json(args.run/"worker.dp0.json")["waves"]
    other=read_json(args.run/"worker.dp1.json")["waves"]
    events=[json.loads(line) for f in args.run.glob("native*.jsonl") for line in f.open()]
    stages=[e for e in events if "stage" in e]
    rows=[]
    for wave in waves:
        if wave["repeat"]<0 or not wave["arm"].startswith("LP_"):continue
        reference=next(w for w in waves if w["arm"]=="FULL" and w["repeat"]==wave["repeat"]
                       and w['modality']==wave['modality'])
        ai,bi=reference["wave"],wave["wave"]
        hidden=errors(torch.load(args.run/f"hidden.w{ai}.dp0.pt",weights_only=True),
            torch.load(args.run/f"hidden.w{bi}.dp0.pt",weights_only=True))
        a=torch.load(args.run/f"logits.w{ai}.dp0.pt",weights_only=True).float()
        b=torch.load(args.run/f"logits.w{bi}.dp0.pt",weights_only=True).float()
        logits=errors(a,b)
        pa,pb=a.softmax(-1),b.softmax(-1);m=(pa+pb)/2
        jsd=float((torch.where(pa>0,pa*(pa.clamp_min(1e-30).log()-m.clamp_min(1e-30).log()),0).sum()
                   +torch.where(pb>0,pb*(pb.clamp_min(1e-30).log()-m.clamp_min(1e-30).log()),0).sum())/2)
        actual=[wave['emitted'].get(str(r).split('-')[0]) for r in wave['injection_ids']]
        expected=[reference['emitted'].get(str(r).split('-')[0]) for r in reference['injection_ids']]
        continuation_valid=(len(actual)==len(expected)==1 and actual[0] is not None
                            and expected[0] is not None and len(actual[0])>=2)
        current=[e for e in stages if e["wave"]==bi]
        semantics=True
        ordering=[]
        for rank in range(4):
            local=sorted([e for e in current if e["ep_rank"]==rank],key=lambda e:e["stage"])
            groups=current[0]["groups"] if current else 0
            semantics &= bool(local) and [e["stage"] for e in local]==list(range(groups))
            semantics &= [tuple(e["range"]) for e in local]==partition(groups) if groups else False
            semantics &= all(len(e["layers"])==48 and all(x["decode_rows"]==4 for x in e["layers"]) for e in local)
            ordering.append([e["range"] for e in local])
            if rank<2:
                semantics &= all(e["prefill_rows"]==wave["prompt_tokens"] for e in local)
        semantics &= all(ordering[0]==o for o in ordering)
        emitted_every_iteration=True
        for rank,source in ((0,wave),(2,next(w for w in other if w['wave']==bi))):
            resident={str(r).split('-')[0] for r in source['resident_ids']}
            local=[e for e in current if e['ep_rank']==rank]
            for e in local:
                outs=source['steps'][e['iteration']]['outputs']
                tokens={o['id']:o['tokens'] for o in outs if o['id'] in resident}
                emitted_every_iteration &= set(tokens)==resident and all(len(t)==1 for t in tokens.values())
        semantics &= emitted_every_iteration
        refs={ (e['layer'],e['ep_rank']):e for e in events
               if e.get('diagnostic')=='route_identity' and e.get('wave')==ai }
        route=[e for e in events if e.get('diagnostic')=='route_identity' and e.get('wave')==bi]
        # SP partitions source rows across TP ranks; compare each same source
        # rank's exact selected expert IDs/weights, not the receiver bank.
        route_identity=(all((e['layer'],e['ep_rank']) in refs and
                        all(e[k]==refs[(e['layer'],e['ep_rank'])][k]
                            for k in ('rows','ids_sha256','weights_sha256')) for e in route)
                        and len(route)==96) if refs else None
        ia=next((e for e in events if e.get('diagnostic')=='prefill_input_identity' and e.get('wave')==ai and e['ep_rank']==0),None)
        ib=next((e for e in events if e.get('diagnostic')=='prefill_input_identity' and e.get('wave')==bi and e['ep_rank']==0),None)
        input_identity=all(ia[k]==ib[k] for k in ('token_count','ids_sha256','positions_sha256','embeddings_sha256')) if ia and ib else None
        passed=(hidden["relative_l2"]<=.005 and hidden["min_row_cosine"]>=.9999
                and logits["max_abs"]<=.25 and jsd<=.0001
                and int(a.argmax())==int(b.argmax()) and continuation_valid and actual==expected and semantics
                and route_identity is not False and input_identity is not False)
        rows.append({"arm":wave["arm"],"modality":wave['modality'],"repeat":wave["repeat"],"hidden":hidden,
            "logits":logits,"logits_jsd":jsd,"first_token_equal":int(a.argmax())==int(b.argmax()),
            "continuation8_equal":actual==expected,"stage_semantics":bool(semantics),
            'continuation_tokens_checked':len(actual[0]) if continuation_valid else None,
            "decode_token_every_stage_iteration":bool(emitted_every_iteration),
            "route_id_weight_identity":route_identity,"pass":bool(passed)})
        rows[-1]['prompt_positions_embeddings_identity']=input_identity
    expected_arms={w["arm"] for w in waves if w["arm"].startswith("LP_")}
    enough=all(sum(r["arm"]==arm for r in rows)>=3 for arm in expected_arms)
    result={"status":"PASS" if rows and all(r["pass"] for r in rows) and enough else "FAIL",
        "repeats_per_arm":{arm:sum(r["arm"]==arm for r in rows) for arm in expected_arms},
        "scope":"CONTROLLED_NATIVE_LP_EP4_HARNESS","rows":rows,
        "source_run":str(args.run)}
    atomic_json(args.output,result);print(result["status"],result["repeats_per_arm"])
    if result["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
