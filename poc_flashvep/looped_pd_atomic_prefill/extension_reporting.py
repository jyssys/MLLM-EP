"""Report extension evidence without hiding earlier failures or mode changes."""
from .common import *
from .evidence import status,compare


def section():
    from .reporting import read,table
    budget=read('configs/extension_budget_20261102.json',{})
    text='## User extension: final deadline 2026-11-02 11:00 KST\n\n'
    text+='Ouro and Huginn discovery had priority. The user also requested AR reproducibility work.\n'
    text+='No new GPU job starts after 10:00 KST. Active owned measurements stop by 10:15 KST.\n'
    text+='The remaining time is reserved for evidence review, publication, and resource restoration.\n'
    text+='The original SLOs and bottleneck gates remain unchanged.\n'
    text+='The new native AR execution family has a separate calibration file.\n'
    text+='Its thresholds do not replace the failed default-family thresholds.\n'
    text+='\n### AR reproduction by actual execution setting\n\n'
    pairs=[
        ('Default PD, changed QPS','pilot_qwen8_l8000_o200_cfull_f0p25_v1','pilot_qwen8_l8000_o200_cfull_f0p5_v1'),
        ('Default arithmetic, Eager PD','ext_qwen_default_eager_pd_qualification24_s0_v1','ext_qwen_default_eager_pd_qualification24_s1_v1'),
        ('True BI + Eager PD, changed QPS','ext_qwen_bi_eager_pd_qualification24_s0_v2','ext_qwen_bi_eager_pd_qualification24_s1_v2'),
        ('True BI + Eager, independent N32 repeat','ext_qwen_bi_eager_qualification32_reference_v1','ext_qwen_bi_eager_qualification32_repeat_v1'),
        ('True BI + Eager, C1024','ext_qwen_bi_eager_qualification32_reference_v1','ext_qwen_bi_eager_qualification32_c1024_v1'),
        ('True BI + Eager, C4096','ext_qwen_bi_eager_qualification32_reference_v1','ext_qwen_bi_eager_qualification32_c4096_v1'),
        ('True BI + Eager, PD versus DP2','ext_qwen_bi_eager_qualification32_reference_v1','ext_qwen_bi_eager_qualification32_dp2_v1'),
        ('True BI + Eager, DP2 repeat','ext_qwen_bi_eager_qualification32_dp2_v1','ext_qwen_bi_eager_dp32_full_repeat_v1'),
        ('BI compiled, graph NONE, PD repeat','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_no_graph_qualification32_repeat_v1'),
        ('BI compiled, graph NONE, PD versus DP2','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_no_graph_qualification32_dp2_v1'),
        ('BI compiled, graph NONE, DP2 repeat','ext_qwen_bi_compiled_no_graph_qualification32_dp2_v1','ext_qwen_bi_compiled_no_graph_qualification32_dp2_repeat_v1'),
        ('BI compiled, graph NONE, C4096','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_no_graph_qualification32_c4096_v1')]
    output=[]
    for label,a,b in pairs:
        if status(a)==status(b)=='COMPLETE':
            result=compare(a,b);output.append([label,result['compared']-len(result['failures']),result['compared'],result['passed'],'MEASURED_NATIVE_GPU_OUTPUT'])
        else:output.append([label,None,None,False,'NOT_MEASURED: '+status(a)+' / '+status(b)])
    text+=table(['Setting','Exact sequences','N','Pass','Evidence'],output)
    text+='These tests establish only the stated internal sequence comparisons.\n'
    text+='An N32 PD pass does not prove DP2 parity, original-HF equivalence, or downstream accuracy.\n'
    text+='Later performance runs also have an exact common-input audit. A new divergence blocks that execution family.\n'
    text+='\nThe first extension BI-labelled v1 trials actually used `ar_batch_invariant=False`.\n'
    text+='The configuration constructor overwrote the explicit request. A regression test detected this bug.\n'
    text+='The constructor now preserves an explicit flag. New v2 namespaces record the effective native flag.\n'
    text+='The incorrectly labelled v1 data remain preserved. They cannot prove that BI failed.\n'
    text+='Earlier genuine BI failures also remain preserved. This correction does not erase those failures.\n'
    text+='See `state/ar_extension_BI_configuration_correction.json` for the affected configurations.\n'
    text+='\n### Numerical probes and their limits\n\n'
    text+='Evidence type: MEASURED_OPERATOR_MICROBENCHMARK and MEASURED_COLD_PREFIX_PROBE.\n'
    text+='The probes use the actual Qwen checkpoint and captured operands. They do not use a proxy model.\n'
    text+='Default arithmetic changes some tested O/Down outputs when only GEMM row geometry changes.\n'
    text+='True BI v2 produces exact selected projection rows in the tested M=1/2/3/4/8/16/32/256 cases.\n'
    text+='This is an operator result, not proof of complete incremental-Decode invariance.\n'
    text+='The cold common-prefix probe selects the same final target rows in all tested batch orders.\n'
    text+='Its two leading BF16 logits are both 33.75. The recorded margin is zero.\n'
    text+='That probe does not reproduce the incremental-Decode divergence.\n'
    text+='The first globally divergent operation remains NOT_AVAILABLE unless a later targeted trace identifies it.\n'
    text+='\n### Qualified Looped runtime controls\n\n'
    controls=read('state/extension_runtime_controls_20261102.json',{})
    text+=table(['Model','Family','Input','Stage','N8 state/output gate','Tuned chunk gate','Reference file'],[
        [v.get('model'),v.get('family'),v.get('length'),v.get('stage'),v.get('gate',{}).get('passed'),
            v.get('chunk_gate_result',{}).get('passed'),v.get('capacity_reference_file')] for v in controls.values()])
    text+='The Huginn controls use the existing async path and graph-row limits of four or eight.\n'
    text+='The fixed-depth model and numerical projection fixes remain unchanged.\n'
    text+='Ouro also tests larger P budgets and a smaller legal source KV pool.\n'
    text+='A graph-row limit is not a claim that every physical GEMM has that row count.\n'
    text+='Ouro 16384/32 is a prospectively declared extension cell. The earlier 16384/100 cell remains preserved.\n'
    text+='Independent service references remain specific to the execution family. Their costs are not pooled.\n'
    text+='\nThe native connector allocates transfer slots separately from compute slots.\n'
    text+='P transfer slots equal P compute slots plus `max_draining_requests`, which defaults to eight.\n'
    text+='D transfer slots also include `max_receiving_requests`, which defaults to 32.\n'
    text+='Therefore a finite optimal P batch is not the same variable as the native transfer-slot count.\n'
    text+='KV reservations can bind before either slot limit. The report keeps those refusals separate.\n'
    wire=read('state/extension_transport_20261102.json',{})
    text+='\n### Native wire-credit controls after workspace qualification\n\n'
    text+=table(['Model','Stage','Exact state/output gates','Pilot runs'],[
        [m,v.get('stage'),{k:r.get('passed') for k,r in v.get('gates',{}).items()},len(v.get('pilots',[]))] for m,v in wire.items()])
    text+='These controls change only supported native credit limits. They do not create a new transport protocol.\n'
    text+='The 64-MiB fragment limit, payload values, and model computation remain unchanged.\n'
    text+='\n### Independent confirmation and deadline limits\n\n'
    confirmations=read('state/extension_confirmation_20261102.json',{})
    text+=table(['Model','Stage','Selected pilot','Repeat N32 gate','Chunk N32 gate','Planned final runs'],[
        [m,v.get('stage'),v.get('choice',{}).get('run',{}).get('name'),v.get('repeat_gate',{}).get('passed'),
            v.get('tuned_gate',{}).get('passed'),len(v.get('finals',[]))] for m,v in confirmations.items()])
    text+='All screened candidates remain in the selection JSON files. The selected N24 run is exploratory.\n'
    text+='A final TRUE verdict still requires three independent starts, actual P kernels, and finite native D-margin validation.\n'
    text+='A queued final run or an independent capacity ratio cannot satisfy this gate.\n'
    text+='Missing or deadline-truncated conditions remain NOT_MEASURED or NOT_RUN_USER_11H_DEADLINE.\n'
    write(ROOT/'extension_evidence_summary_20261102.md',text)
    return text


if __name__=='__main__':section()
