"""Validate native P graph reuse before any serving expansion."""
from .common import *
from .prepare import base, put
from .run import execute
from .campaign import gate


def main():
    ledger = []
    for model, length in [('ouro', 4096), ('huginn', 2560)]:
        reference = None
        for label, enabled in [('eager', False), ('native_graph', True)]:
            cfg = base(model, f'prefill_graph_{model}_{label}_gate8',
                       length=length, cap=8, phase='correctness',
                       verify_transfer=True, state_probe=True,
                       projection_graph_rows=0, P_slots=2, kv_blocks=16384,
                       token_budget=length*2, timeout=600)
            if enabled:
                cfg['P_prefill_graph_rows'] = length*2
                cfg['P_prefill_graph_shapes'] = 4
            name = put(cfg, 8)
            write(ROOT/'state/prefill_graph_controls.json',
                  dict(stage='CORRECTNESS', name=name, time=time.time()))
            okay = execute(name) and gate(name, reference)
            record = dict(model=model, name=name, passed=okay)
            if enabled and okay:
                path = ROOT/'raw'/name/'native_prefill_graph_counts.jsonl'
                counts = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                record['graph_counts'] = counts
                record['actual_graph_replay'] = any(v['replays'] > 0 for v in counts)
                okay = okay and record['actual_graph_replay']
                record['passed'] = okay
            ledger.append(record)
            write(ROOT/'state/prefill_graph_control_results.json', ledger)
            if not okay:
                break
            reference = name
    write(ROOT/'state/prefill_graph_controls.json',
          dict(stage='CONTROLS_COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/prefill_graph_controls.json',
              dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
