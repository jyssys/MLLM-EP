"""CPU-only validation: delayed frontend polling cannot inflate core-ready E2E."""
from types import SimpleNamespace
from run_ep_serving import record_token_timing


def main():
    rec={"submitted_monotonic":100.,"token_timestamps":[],"frontend_token_timestamps":[]}
    metrics=SimpleNamespace(last_token_ts=100.2,first_token_ts=100.2,
                            queued_ts=100.05,scheduled_ts=100.06)
    out=SimpleNamespace(metrics=metrics,outputs=[SimpleNamespace(token_ids=[7])],finished=False)
    assert record_token_timing(rec,out,100.7)==100.2
    metrics.last_token_ts=100.3
    out.outputs[0].token_ids=[7,8]
    out.finished=True
    assert record_token_timing(rec,out,100.8)==100.3
    assert rec["token_timestamps"]==[100.2,100.3]
    assert abs(rec["frontend_poll_delay_ms"]-500)<1e-8
    metrics.last_token_ts=999.
    assert rec["engine_metrics"]["last_token_ts"]==100.3
    rec2={"submitted_monotonic":100.,"token_timestamps":[],"frontend_token_timestamps":[]}
    metrics.last_token_ts=100.5
    out.outputs[0].token_ids=[7,8,9]
    record_token_timing(rec2,out,100.8)
    assert rec2['token_timestamps'][0]==100.2
    assert rec2['coalesced_token_observations']==1
    print("PASS: core-ready/receipt separated; metrics snapshotted; CPU only")


if __name__=="__main__":main()
