"""Apply the frozen task scorers to untouched Natural output token ledgers."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

from ..mllm_epd_prefill_heavy_token_split import quality_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scorer-root",type=Path,default=Path(
        "poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124"))
    args=p.parse_args()
    quality_audit.ROOT=args.scorer_root
    scoring=quality_audit.scorers()
    tk=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows={row["request_id"]:row for row in json.loads(args.manifest.read_text())["requests"]}
    sources, score_rows = {}, []
    for path in args.runs:
        run=path.parent.name
        data=json.loads(path.read_text())
        if data["cap"]!=2048 or data["workload"]!="realistic":
            raise ValueError("Only original image prompts with Natural decoding are scored")
        by_task=defaultdict(list)
        for request in data["requests"]:
            original=rows[request["id"]]
            score={"score":None,"complete_task_scorer":False,"official_full_pipeline":"NOT_RUN"}
            if request["outcome"] in ("eos","cap"):
                text=tk.decode(request["output_ids"],skip_special_tokens=True)
                with redirect_stdout(io.StringIO()):
                    score=quality_audit.score_request(original,text,scoring)
            row={"run":run,"source":original["source"],"request_id":request["id"],"ordinal":request["ordinal"],
                 "finish":request["outcome"],"score":score["score"],"scorer":score.get("metric"),
                 "complete_task_scorer":score["complete_task_scorer"],
                 "official_pipeline":score.get("official_full_pipeline","SCORED")}
            score_rows.append(row)
            by_task[row["source"]].append(row)
        sources[run]=[{"task":task,"planned":len(items),"scored":len(points),
                       "mean":mean(points) if points else None,
                       "official_pipeline":sorted({d["official_pipeline"] for d in items})}
                      for task,items in sorted(by_task.items())
                      for points in [[d["score"] for d in items if d["score"] is not None]]]
    with args.out.open("x") as stream:
        json.dump({"scope":"Local pinned task scorers. MMLongBench-Doc external extraction is NOT_RUN.",
                   "source_receipts":scoring[3],"summaries":sources,"request_scores":score_rows},stream,indent=2)
        stream.write("\n")


if __name__=="__main__":
    main()
