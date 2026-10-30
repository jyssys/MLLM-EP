"""Join diagnostic readiness observations, submission, and completion tickets."""
from collections import defaultdict
import json

from .common import ROOT
from .analyze import save


def main():
    records = []
    quanta = []
    costs = []
    for folder in sorted((ROOT / 'raw').iterdir()):
        contract_path = folder / 'serving_contract.json'
        if not contract_path.exists():
            continue
        contract = json.loads(contract_path.read_text())
        if not contract.get('diagnostic'):
            continue
        for path in sorted(folder.glob('transfer_events_*.jsonl')):
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            callbacks = [row for row in values if row['kind'] == 'prefill_quantum_callback']
            costs += [dict(row, run=folder.name, model=contract['model'], arm=contract['arm'])
                      for row in values if row['kind'] == 'transport_submission_summary']
            quanta += [dict(row, run=folder.name, model=contract['model'], arm=contract['arm']) for row in callbacks]
            groups = defaultdict(dict)
            for value in values:
                if 'sequence' in value:
                    groups[value['tid'], value['sequence']][value['kind']] = value
            for (tid, sequence), events in groups.items():
                ready = events.get('ready_first_observed', {})
                submit = events.get('submitted', {})
                done = events.get('completion_observed', {})
                callback = next((row for row in callbacks if row['tid'] == tid
                                 and row['sequence_start'] <= sequence < row['sequence_end']), {})
                row = dict(run=folder.name, model=contract['model'], arm=contract['arm'],
                           pid=(ready or submit or done)['pid'], transfer_id=tid, sequence=sequence,
                           bytes=(submit or done or ready).get('bytes'),
                           ready_first_observed_s=ready.get('time', 'NOT_MEASURED'),
                           submit_observed_s=submit.get('time', 'NOT_MEASURED'),
                           completion_observed_s=done.get('time', 'NOT_MEASURED'),
                           metadata_and_issue_us=submit.get('metadata_and_issue_us', 'NOT_MEASURED'),
                           callback_chunk_start=callback.get('chunk_start', 'NOT_MEASURED'),
                           callback_chunk_complete=callback.get('chunk_complete', 'NOT_MEASURED'),
                           callback_prompt_complete=callback.get('prompt_complete', 'NOT_MEASURED'),
                           request_id=callback.get('request_id', 'NOT_MEASURED'),
                           evidence='MEASURED_DIAGNOSTIC_HOST_OBSERVATIONS',
                           limitation='Polling observations are not exact kernel or DMA timestamps. Durations overlap. Not clean serving.')
                row['ready_observed_to_submit_ms'] = 1000 * (submit['time'] - ready['time']) if ready and submit else 'NOT_MEASURED'
                row['submit_ticket_to_completion_observed_ms'] = 1000 * (done['time'] - done['actual_submit']) if done else 'NOT_MEASURED'
                records.append(row)
    save('diagnostic_transfer_timing.csv', records or [dict(evidence='NOT_MEASURED', reason='No diagnostic transfer records.')])
    save('diagnostic_prefill_callbacks.csv', quanta or [dict(evidence='NOT_MEASURED', reason='No diagnostic prefill callbacks.')])
    save('diagnostic_submission_costs.csv', costs or [dict(evidence='NOT_MEASURED', reason='No diagnostic submission-cost summary.')])
    return records


if __name__ == '__main__':
    main()
