"""Archive closed numeric records. Do not read an active worker's partial output."""
import gzip
import json
import time
from .common import ROOT,digest,write

def main():
    target=ROOT/'archives/checkpoint_completed_numeric.jsonl.gz'
    target.parent.mkdir(parents=True,exist_ok=True)
    sources=[];completed=[]
    prefixes=('ep_prefill_','ep_decode_','generation_','generic_generation_',
        'native_forward_','native_generation_','shape_fidelity','metadata.',
        'ownership.','fidelity.','official_summary.','complete.',
        'nll_sequences.','nll_summary.','paired_','verify_')
    with gzip.open(target,'wt') as output:
        for directory in sorted((ROOT/'raw').iterdir()):
            if not directory.is_dir() or not (directory/'complete.json').is_file():continue
            completed.append(directory.name)
            paths=[p for p in sorted(directory.iterdir()) if p.is_file()
                and p.suffix in ['.json','.jsonl'] and p.name.startswith(prefixes)]
            native_summary=directory/'official/results.native.json'
            if native_summary.is_file():paths.append(native_summary)
            for path in paths:
                data=([json.loads(line) for line in path.read_text().splitlines()]
                    if path.suffix=='.jsonl' else json.loads(path.read_text()))
                relative=path.relative_to(ROOT).as_posix()
                output.write(json.dumps(dict(source=relative,data=data))+'\n')
                sources.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path)))
    states={name:json.loads((ROOT/name).read_text()) for name in
        ['research_state.json','post_state.json','attribution_campaign_state.json','finalization_state.json']}
    write(ROOT/'quality/checkpoint_evidence_manifest.json',dict(
        status='IN_PROGRESS_CHECKPOINT',time=time.time(),
        archive=target.relative_to(ROOT).as_posix(),archive_sha256=digest(target),
        completed_attempts=completed,sources=sources,campaign_states=states,
        scope='Closed numeric files from completed attempts. Active and failed attempts are not counted as complete.',
        raw_storage='All original raw files remain local. Full benchmark prompts are not republished.',
        limitation='This checkpoint does not contain the pending strong-generic matrix or whole-E2E component percentages.'))
    print(dict(completed_attempts=len(completed),archived_numeric_files=len(sources),
        archive_bytes=target.stat().st_size),flush=True)

if __name__=='__main__':main()
