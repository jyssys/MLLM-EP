"""Append-only template-boundary repair; no cohort reselection or label use."""
import gzip
import json
from transformers import AutoTokenizer
from poc_agentic_ep.contract import ROOT,TARGET,write,sha,observation_frontier


def main():
    path=ROOT/'results/frozen_manifest_v2.json'
    if path.exists():raise FileExistsError(path)
    previous=ROOT/'results/frozen_manifest.json'
    manifest=json.loads(previous.read_text())
    tokenizer=AutoTokenizer.from_pretrained(TARGET,local_files_only=True)
    messages={r['trajectory_id']:r['messages'] for r in json.load(gzip.open(ROOT/'results/frozen_messages.json.gz','rt'))}
    for row in manifest['rows']:
        history=messages[row['trajectory_id']]
        endpoint=[i for i,m in enumerate(history) if m['role']=='assistant'][row['turn']]
        frontier=observation_frontier(history[:endpoint],tokenizer,row['prompt_token_ids']) if row['turn'] else 0
        row['current_observation_frontier']=frontier
        row['exclusion_start']=max(row['common_prompt_prefix'],row['previous_completed_prefix'],frontier)
        row['new_suffix_tokens']=row['prompt_tokens']-row['exclusion_start']
    manifest['supersedes_exclusion_only']=dict(path=str(previous),sha256=sha(previous),
        cause='Qwen3 template adds empty think block for last assistant but removes it when assistant becomes history; plain completed-context LCP can stop before repeated assistant content',
        repair='current-rendering last observation span conservatively excludes all prior semantic messages in addition to exact LCP',
        validation_state='two calibration smoke turns only; held-out routing has not been observed; cohort/input IDs/continuation unchanged')
    manifest['exclusion']='max(previous prompt LCP, previous completed LCP, current-rendering last-observation frontier, actual KV-cache prefix); no historical content in primary analysis'
    write(path,manifest)
    print(json.dumps(dict(turns=len(manifest['rows']),new_suffix_total=sum(r['new_suffix_tokens'] for r in manifest['rows']),
        minimum_new_suffix=min(r['new_suffix_tokens'] for r in manifest['rows']))))


if __name__=='__main__':main()
