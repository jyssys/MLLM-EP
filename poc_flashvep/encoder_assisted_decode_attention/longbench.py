"""Freeze a modest natural-text cohort from the publisher's original LongBench."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from transformers import AutoTokenizer


SHA = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
TASKS = ("qasper", "hotpotqa", "narrativeqa", "gov_report")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def put(path,data):
    with path.open("x") as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False)
        stream.write("\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip",type=Path,required=True)
    p.add_argument("--templates",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    if digest(args.zip)!=SHA:
        raise ValueError("Original publisher archive SHA mismatch")
    templates=json.loads(args.templates.read_text())
    tk=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    chosen=[]
    with zipfile.ZipFile(args.zip) as archive:
        for task in TASKS:
            name=next(item for item in archive.namelist() if Path(item).name==task+".jsonl")
            candidates=[]
            with archive.open(name) as data:
                for value in map(json.loads,data):
                    text=templates[task].format(**value)
                    prompt=tk.apply_chat_template([{"role":"user","content":text}], tokenize=False,
                                                  add_generation_prompt=True)
                    n=len(tk.encode(prompt,add_special_tokens=False))
                    if 2048 <= n <= 28672:
                        ident=str(value["_id"])
                        candidates.append({"request_id":f"longbench/{task}/{ident}","source":task,
                                           "study_source":"LongBench","image_count":0,"visual_tokens":0,
                                           "total_input_tokens":n,"prompt":prompt,
                                           "question":value.get("input",""),"reference_answers":value["answers"],
                                           "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest()})
            if len(candidates)<32:
                raise ValueError(f"Task {task} lacks the declared frozen sample")
            candidates.sort(key=lambda row:hashlib.sha256(("eada-nov27-v1:"+row["request_id"]).encode()).digest())
            chosen.extend(candidates[:32])
    put(args.out,{"requests":chosen})
    put(args.receipt,{"publisher":"zai-org/LongBench","revision":REVISION,"archive_sha256":SHA,
                      "templates_sha256":digest(args.templates),"manifest_sha256":digest(args.out),
                      "task_counts":dict(Counter(row["source"] for row in chosen)),
                      "rules":"32 per task, 2048–28672 original templated Qwen tokens, SHA selector eada-nov27-v1",
                      "rows":[{k:row[k] for k in ("request_id","source","total_input_tokens","prompt_SHA256")} for row in chosen]})


if __name__=="__main__":
    main()
