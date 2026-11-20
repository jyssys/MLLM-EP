"""Update derived evidence when a completed input set changes."""
import contextlib
import io
import json
import time
from .common import ROOT,write
from .monitor import snapshot


def main():
    path=ROOT/'state/analysis_watch_state.json'
    state=json.loads(path.read_text()) if path.exists() else {};actions=[];errors=[]
    def run(key,inputs,function):
        existing=[p for p in inputs if p.exists()]
        if not existing:return
        signature=[(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in existing]
        signature=json.loads(json.dumps(signature))
        if state.get(key)==signature:return
        try:
            with contextlib.redirect_stdout(io.StringIO()):function()
            state[key]=signature;actions.append(key)
        except (json.JSONDecodeError,FileNotFoundError):return
        except Exception as error:errors.append(dict(action=key,error=repr(error)))
    from . import profile_analysis,credit_decomposition,statistics,replay_goodput,causal_replay
    run('profiles_v2',list((ROOT/'analysis').glob('*profile_triplets.json'))+
        list((ROOT/'analysis').glob('natural_Last4_*_profile_triplet.json'))+
        list((ROOT/'analysis').glob('sparse_P_profile_fallback_v1.json')),profile_analysis.main)
    run('credit',[(ROOT/'analysis/credit_sweep_v1.json'),ROOT/'analysis/serial_online_causal_control_v1.json'],credit_decomposition.main)
    run('statistics',[ROOT/'analysis/last12_native_reference_repair.json',ROOT/'analysis/full200_allcuts_confirmation_v1.json'],statistics.main)
    if (ROOT/'state/Last8_serial_online_causal_control_complete.json').exists():
        run('last8',[ROOT/'analysis/Last8_serial_online_causal_control_v1.json'],lambda:causal_replay.main(8))
    if (ROOT/'state/heldout_arrival_service_replay_complete.json').exists():
        run('heldout_goodput',[ROOT/'analysis/heldout_arrival_service_replay_v1.json'],replay_goodput.main)
    historical=ROOT/'analysis/historical_runtime_bridge_v1.json'
    if historical.exists():
        from .historical_bridge_analysis import main as historical_analysis
        run('historical_bridge',[historical],historical_analysis)
        from .destination_chain import main as destination_chain
        run('destination_chain',[historical],destination_chain)
    from .bridge_statistics import main as bridge_statistics
    run('bridge_statistics',[historical,ROOT/'analysis/historical_block16_confirmation_v1.json',
        ROOT/'analysis/historical_block16_extra_holdout_v1.json',ROOT/'analysis/coalescing_paired_repetition_v1.json'],bridge_statistics)
    write(path,state)
    row=snapshot();row.update(analysis_actions=actions,analysis_errors=errors)
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
