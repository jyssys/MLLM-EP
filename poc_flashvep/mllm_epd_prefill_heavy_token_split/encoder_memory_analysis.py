"""Report observed memory fit, not encoder-safe compute or maximum capacity."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write


def main():
    preserve()
    path=ROOT/'analysis/encoder_LLM_MEMORY_C16.json'
    value=json.loads(path.read_text())
    assert value['status']=='OBSERVED_CO_RESIDENCY_FIT'
    features=value['encoder_feature_comparison']
    assert features and all(r['strict'] and r['finite'] for r in features)
    rows=[]
    for label,key in [('E loaded','initial_E'),('E after real encodes','E_after_encode'),
                      ('E + full model loaded','with_LLM_E'),('Co-resident after real encodes','final_E')]:
        state=value[key]
        rows.append(dict(stage=label,device_total_bytes=state['device_total_bytes'],
            device_used_bytes=state['device_total_bytes']-state['device_free_bytes'],
            device_free_bytes=state['device_free_bytes'],E_allocated_bytes=state['allocated_bytes'],
            E_reserved_bytes=state['reserved_bytes'],E_peak_allocated_bytes=state['maximum_allocated_bytes'],
            evidence='ISOLATED_MEMORY_MEASUREMENT'))
    base=ROOT/'figures/encoder_LLM_memory_C16_v0'
    with base.with_suffix('.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([r['stage'] for r in rows],[r['device_used_bytes']/1024**3 for r in rows])
    ax.axhline(rows[0]['device_total_bytes']/1024**3,color='black',ls=':',label='Observed device total')
    ax.set_ylabel('Device used memory (GiB)')
    ax.set_title('ISOLATED MEMORY | Original encoder + extra full Qwen2.5-VL model')
    ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.text(.02,.01,'One initialization. 3 real input stresses. No additional prefill or online SLO test.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1])
    for suffix in ['png','pdf']:fig.savefig(base.with_suffix('.'+suffix),dpi=180)
    plt.close(fig)
    record=dict(time=time.time(),raw_SHA256=digest(path),status='OBSERVED_CO_RESIDENCY_FIT',
        requests=len(value['before_requests']),strict_images=len(features),rows=rows,
        helper_model_parameter_bytes=value['additional_full_model']['model_parameter_logical_bytes'],
        helper_LLM_parameter_bytes=value['language_model_parameter_bytes'],
        helper_duplicate_vision_bytes=value['additional_full_model']['model_parameter_bytes_by_prefix']['visual'],
        helper_KV_storage_bytes=value['helper_KV_storage_bytes'],
        encode_safe_headroom='NOT_MEASURED',extra_prefill_work='NOT_RUN')
    write(ROOT/'analysis/encoder_LLM_memory_C16_summary_v0.json',record)
    preserve()
    print('MEMORY_SUMMARY',record['requests'],record['strict_images'],rows[-1]['device_free_bytes'],flush=True)


if __name__=='__main__':main()
