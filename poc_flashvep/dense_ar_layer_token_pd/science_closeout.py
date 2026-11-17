"""Close experiment admission after the planned confirmation finishes."""
import json
import time
from .common import ROOT,preserve,write


def main():
    complete=ROOT/'state/candidate_confirmation_complete.json'
    assert complete.exists(),'The confirmation controller has not finished'
    record=json.loads(complete.read_text())
    progress=json.loads((ROOT/'state/progress.json').read_text())
    target=ROOT/'state/science_closeout.json'
    assert not target.exists(),'Preserve the original science closeout'
    preserve();now=time.time()
    marker=ROOT/'state/CAMPAIGN_CLOSED.json'
    if not marker.exists():
        write(marker,dict(time=now,status='GPU_SCIENCE_COMPLETE',
            reason='Planned confirmation has finished. Preserve all failed or omitted arms. Close admission for analysis, verified evidence publication and approved burner restoration. This does not mark all research goals as achieved.'))
    write(target,dict(time=now,start_epoch=progress['start_epoch'],
        science_wall_seconds=now-progress['start_epoch'],planned_deadline_epoch=progress['deadline_epoch'],
        completed_confirmation_entries=record['completed_entries'],
        expected_confirmation_entries=42,
        scope='Wall-clock research time includes GPU startup, correctness capture, CPU analysis and waiting for the shared lease. It is not exclusive GPU compute. Research gates remain separate from campaign completion.'))
    snapshot=ROOT/'state/progress_initial_snapshot.json'
    if not snapshot.exists():
        write(snapshot,progress)
    progress.update(status='CAMPAIGN_CLOSED_WITH_LIMITATIONS',phase='FINAL_EVIDENCE_AND_PUBLICATION',
        science_closeout_epoch=now,
        research_gate_scope='TP2 strict baseline and dynamic contribution must be read from final independent-start evidence. A closed campaign is not an all-goals success.',
        publication_scope='Remote SHA is verified after the commit. Read the local publication receipt and final user response.')
    write(ROOT/'state/progress.json',progress)
    preserve();print('GPU_SCIENCE_CLOSED',record['completed_entries'],flush=True)


if __name__=='__main__':
    main()
