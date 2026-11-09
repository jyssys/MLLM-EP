"""Publish a compact progress inventory from immutable worker receipts."""
from .common import *
from .qualification import qualify
from .analyze import metrics


def main():
    configure();rows=[]
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json'),key=lambda p:p.stat().st_mtime):
        name=path.parent.name;receipt=json.loads(path.read_text())
        row=dict(name=name,complete=receipt.get('complete'),elapsed_s=receipt['elapsed_seconds'])
        if row['complete']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            ref=f'n32_{cfg["model"]}_native_full_state32' if cfg['count']==32 else f'a_{cfg["model"]}_native_state8' if cfg['count']==8 else None
            if ref and not cfg['state_probe']:ref=None
            try:
                q=qualify(name,ref);m=metrics(name)
                row.update(passed=q['passed'],offloaded=q['offload']['requests'],
                    loops=q['offload']['accepted_loop_counts'],suffix=q['offload']['accepted_suffix_tokens'],
                    diversity=q['offload']['adaptive_diversity'],TTFT_p95_ms=m['TTFT_p95_ms'],
                    ITL_p99_ms=m['ITL_p99_ms'],TPOT_p95_ms=m['TPOT_p95_ms'])
            except Exception as error:row['analysis_error']=repr(error)
        rows.append(row)
    write(ROOT/'progress_inventory.json',rows)
    print(json.dumps(rows[-24:],indent=2))


if __name__=='__main__':main()
