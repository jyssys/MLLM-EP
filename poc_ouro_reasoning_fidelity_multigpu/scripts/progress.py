"""Read saved progress without touching model processes or GPU state."""
import datetime
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    state = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    quality = ROOT/'results/quality_v1/progress.json'
    if quality.exists():
        state['native_quality_last_wave'] = read(quality)[-1]
    reference = ROOT/'results/hf_prompt_corroboration_v1/reference_completions.json'
    state['reference_saved'] = [] if not reference.exists() else [
        {k: row[k] for k in ['problem_id', 'arm', 'generated_tokens', 'finish_reason']}
        for row in read(reference)]
    for name in ['restart_same_history_default_v1', 'restart_same_history_default_v2',
                 'restart_same_history_MP0_v1', 'restart_same_history_MP0_v2']:
        history = ROOT/'results'/name/'request_history.json'
        if history.exists():
            state[name] = dict(saved_client_calls=len(read(history)))
    for tp in [1, 2, 4]:
        path = ROOT/f'results/exploratory_B1_TP{tp}_v1/natural_requests.json'
        if path.exists():
            state[f'exploratory_B1_TP{tp}'] = dict(saved_requests=len(read(path)))
    state['bounded_followup_complete'] = (ROOT/'results/bounded_followup_execution.json').exists()
    print(json.dumps(state, sort_keys=True))


if __name__ == '__main__':
    main()
