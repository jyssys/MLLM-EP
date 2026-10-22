"""Freeze public conversational and article prompts before serving comparisons."""

import hashlib
import json
import urllib.request
from .common import ROOT, MODEL, write


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    return json.loads(data), dict(
        url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
    )


def main():
    file = ROOT / "traffic_workload_manifest.json"
    assert not file.exists(), "Frozen traffic inputs cannot be overwritten"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, local_files_only=True
    )
    alpaca, alpaca_source = fetch(
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    )
    article_rows, article_source = fetch(
        "https://datasets-server.huggingface.co/rows?dataset=ccdv%2Farxiv-summarization&config=section&split=test&offset=0&length=40"
    )
    cases = []
    for i, row in enumerate(alpaca[:96]):
        content = row["instruction"]
        if row["input"]:
            content += "\n" + row["input"]
        content += "\nGive a concise, complete answer."
        text = tokenizer.apply_chat_template(
            [dict(role="user", content=content)],
            tokenize=False,
            add_generation_prompt=True,
        )
        cases.append(
            dict(
                id=f"alpaca_{i}",
                domain="conversation_proxy",
                split="calibration" if i < 16 else "heldout",
                input_ids=tokenizer.encode(text),
                text=text,
                source_row=i,
            )
        )
    for i, item in enumerate(article_rows["rows"]):
        article = item["row"]["article"]
        article_ids = tokenizer.encode(article)
        template = tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content="Summarize the following scientific article excerpt in two complete sentences.\n<ARTICLE>\nSummary:",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        left, right = template.split("<ARTICLE>")
        prefix, suffix = tokenizer.encode(left), tokenizer.encode(right)
        for length in (2048, 8192):
            count = length - len(prefix) - len(suffix)
            if len(article_ids) < count:
                continue  # Do not repeat short text to manufacture a long article.
            ids = prefix + article_ids[:count] + suffix
            cases.append(
                dict(
                    id=f"arxiv_{i}_P{length}",
                    domain="article_excerpt",
                    split="calibration" if i < 8 else "heldout",
                    input_ids=ids,
                    text=tokenizer.decode(ids),
                    source_row=item["row_idx"],
                    prompt_bin=length,
                    source_article_tokens=len(article_ids),
                )
            )
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert stop_id is not None and stop_id != tokenizer.unk_token_id
    record = dict(
        cases=cases,
        sources=[alpaca_source, article_source],
        model_revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        chat_template=tokenizer.chat_template,
        EOS_token_id=tokenizer.eos_token_id,
        frontend_stop_token_ids=[stop_id],
        stop_contract="Native EOS or naturally generated chat-turn end. Frontend cancels only after the stop token is actually observed. Stop token counts in output length.",
        generation=dict(
            temperature=0.0,
            max_loops=4,
            exit_threshold=1.0,
            complete_lane_cap=512,
            extension_cap=4096,
            bounded_lane_cap=128,
        ),
        selection="Alpaca first96 input-only rows. arXiv test first40 rows, retain naturally long-enough article excerpts. No output-based filtering.",
    )
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    write(file, record)
    print("FROZEN_TRAFFIC", len(cases), "cases", stop_id)


if __name__ == "__main__":
    main()
