"""Report extension evidence without hiding earlier failures or mode changes."""
from .common import *
from .evidence import status,compare


def section():
    from .reporting import read,table,prose_blocks
    budget=read('configs/extension_budget_20261102.json',{})
    text='## User extension: final deadline 2026-11-02 11:00 KST\n\n'
    text+='Ouro and Huginn discovery had priority. The user also requested AR reproducibility work.\n'
    text+='No new GPU job starts after 10:05 KST. Active owned measurements stop by 10:15 KST.\n'
    text+='At 09:47 KST, the start cutoff moved five minutes later. The earlier budget file remains preserved.\n'
    text+='The GPU stop, final deadline, correctness gates, and frozen SLOs did not change.\n'
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
        ('BI compiled, graph NONE, C4096','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_no_graph_qualification32_c4096_v1'),
        ('BI compiled, P4 startup recheck, Full','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_full_v2'),
        ('BI compiled, P4 startup recheck, C4096','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_c4096_v2'),
        ('BI compiled, P4 corrected-input recheck, Full','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_full_v3'),
        ('BI compiled, P4 corrected-input recheck, C4096','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_c4096_v3'),
        ('BI compiled, P4 corrected-input recheck, DP2','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_dp2_v3'),
        ('BI compiled, P4 corrected-input recheck, DP2 repeat','ext_qwen_bi_compiled_no_graph_qualification32_reference_v1','ext_qwen_bi_compiled_P4_recheck32_dp2_repeat_v3')]
    output=[]
    for label,a,b in pairs:
        if status(a)==status(b)=='COMPLETE':
            result=compare(a,b)
            output.append([label,result['compared']-len(result['failures']) if result['comparison_applicable'] else None,
                result['compared'],result['passed'],'MEASURED_NATIVE_GPU_OUTPUT' if result['comparison_applicable'] else 'NOT_COMPARABLE_INPUT_OR_SEED'])
        else:output.append([label,None,None,False,'NOT_MEASURED: '+status(a)+' / '+status(b)])
    text+=table(['Setting','Exact sequences','N','Pass','Evidence'],output)
    text+='These tests establish only the stated internal sequence comparisons.\n'
    text+='An N32 PD pass does not prove DP2 parity, original-HF equivalence, or downstream accuracy.\n'
    text+='Later performance runs also have an exact common-input audit. A new divergence blocks that execution family.\n'
    recovery=read('state/ar_compiled_recovery_20261102.json',{})
    if recovery.get('P4_recovery'):
        text+='\nThe first compiled QPS pilot failed during P-worker startup, before any request completed.\n'
        text+='The child log reports a CUDA illegal-memory-access error during native kernel warm-up.\n'
        text+='The capacity selector reduced engine request capacity from the qualified four to one.\n'
        text+='A finite optimal batch size does not certify a smaller engine request capacity.\n'
        text+='The new P4 checks restore the earlier qualified request capacity. They do not modify native warm-up or model arithmetic.\n'
        text+='The exact failing GPU operation remains unlocalized. One startup failure does not establish that P1 alone caused it.\n'
        text+='The failed namespace and the earlier family blocker remain preserved. P4 performance uses a separate family label.\n'
        text+='The earlier independent finite-service references remain diagnostic references, not measured P4 sustainable capacity.\n'
        text+='The v2 rechecks selected the default test pool, not the earlier qualification subset.\n'
        text+='All 32 input rows differed. Those original-reference comparisons cannot diagnose numerical failure.\n'
        text+='Corrected v3 checks explicitly select the earlier dataset pool. The invalid comparisons remain preserved.\n'
    if recovery.get('screening_pilots'):
        screens=recovery['screening_pilots']
        text+=f'\nThe bounded AR search defines {len(screens)} coarse screens across three methods and three rate factors.\n'
        text+=f'Completed coarse screens: {sum(status(n)=="COMPLETE" for n in screens)}/{len(screens)}.\n'
        text+=f'The immutable exploratory plan retains {len(recovery.get("pilots",[]))} configurations across all seven requested factors.\n'
        text+='Confirmation takes priority after the coarse grid. Unstarted extra factors and DP2 final repeats remain explicit budget omissions.\n'
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
    text+='The pinned Huginn engine supports Sync execution only. Graph-row limits of four and eight remain supported controls.\n'
    text+='The requested Async path was not active in the earlier successful native-PD trials.\n'
    text+='The inherited launcher forwards the Async flag only for Ouro. Direct Huginn Async construction fails its native guard.\n'
    text+='The correction preserves those actual Sync measurements and retires unstarted unsupported or redundant jobs.\n'
    text+='Explicit Sync v2 namespaces provide the corrected comparisons. No engine guard is removed.\n'
    text+='New `effective_execution_*.json` files record the actual StageEngine flags before measurement.\n'
    text+='Earlier effective Sync attribution is SOURCE_AUDIT, not a recorded startup flag.\n'
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
    text+='The source-credit controls compare fewer draining slots at unchanged compute slots and wire credit.\n'
    text+='A slower P configuration cannot pass by creating a queue. It must retain near-reference P service and actual P kernel saturation.\n'
    text+='\nHuginn D4 uses an explicit preparation-only ACK control.\n'
    text+='Each valid import can release its source allocation before the full D cohort is ready.\n'
    text+='The existing D barrier still blocks all Decode until every import is valid.\n'
    text+='This separates the three-context source pool from the legal four-context D pool.\n'
    text+='Its finite D timing is not a measured Serving speedup.\n'
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
    text+='Ouro also profiles its best N24 candidate at each extension input length before final selection.\n'
    text+='Only an exact matched profile within the original 5% overhead limit can guide that selection.\n'
    text+='Those exploratory profiles do not replace the N32 or N128 gates.\n'
    text+='Ouro and Huginn each completed three center-rate clean starts for both selected chunk settings.\n'
    text+='Extra heavy traces can be omitted after their clean P-service prerequisite fails.\n'
    text+='The queue retains one root-cause trace per chunk, attempted traces, and every service-qualified profile.\n'
    text+='The omitted namespaces remain NOT_RUN_FAILED_STRONG_P_SERVICE_PREREQUISITE in the ledger.\n'
    text+='A final TRUE verdict still requires three independent starts, actual P kernels, and finite native D-margin validation.\n'
    text+='If P compute does not qualify, one small native staggered-D test can still answer the separate D-reserve question.\n'
    text+='Its purpose is D_RESERVE_ONLY_WITHOUT_QUALIFIED_P_COMPUTE. A pass does not establish a Prefill-heavy regime.\n'
    text+='A queued final run or an independent capacity ratio cannot satisfy this gate.\n'
    text+='Missing or deadline-truncated conditions remain NOT_MEASURED or NOT_RUN_USER_11H_DEADLINE.\n'
    retirement=read('state/optional_looped_tail_retirement_20261102.json',{})
    if retirement:
        text+=f'At the final budget review, {len(retirement.get("retired",[]))} unstarted optional Looped tail jobs were omitted.\n'
        text+='Every unstarted tail job used the same priority rule. No active GPU experiment stopped for this budget change.\n'
        text+='The ledger marks them NOT_RUN_EXTENSION_PRIORITY_TIME_BUDGET. Their immutable configurations remain preserved.\n'
    text+='\n### Profiler failures and their scope\n\n'
    stopped=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        watchdog=folder/'profile_watchdog.json';manual=folder/'manual_no_progress_stop.json'
        if not watchdog.exists() and not manual.exists():continue
        path=watchdog if watchdog.exists() else manual
        event=json.loads(path.read_text())
        config_path=ROOT/'configs'/(folder.name+'.json')
        cfg=json.loads(config_path.read_text()) if config_path.exists() else {}
        response_path=folder/'requests.jsonl';completed=0
        if response_path.exists():
            with response_path.open() as incoming:
                completed=sum(not json.loads(line).get('warmup',False) for line in incoming if line.strip())
        result_path=folder/'dispatcher_result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        stopped.append([folder.name,cfg.get('model'),completed,cfg.get('count'),
            event.get('reason',result.get('stop_reason')),result.get('returncode'),str(path.relative_to(ROOT))])
    text+=table(['Profile run','Model','Completed responses','Offered requests','Stop reason','Return code','Evidence'],stopped)
    text+='These runs stopped after response progress ceased or after a bounded profiler limit.\n'
    text+='The registered owned child sessions stopped. Other users\' processes remained unchanged.\n'
    text+='Partial responses and profiler files remain preserved. They do not establish complete-run throughput or kernel saturation.\n'
    text+='Their matched clean runs remain separate evidence. A profiler-associated stall does not identify its root cause.\n'
    text+='The failed Nsight runs do not prove a CUDA, NIXL, or model error.\n'
    ar_profile_budget=read('state/ar_profile_stall_tail_gate_20261102.json',{})
    if ar_profile_budget:
        text+=f'After the near-deadline AR stall, {len(ar_profile_budget.get("omitted",[]))} unstarted AR profiles were omitted.\n'
        text+='Every clean start and attempted trace remains preserved. This omission cannot support a three-profile TRUE claim.\n'
    text+='The CPU process observer also failed once during a /proc exit race. The observer now handles that race.\n'
    text+='Its regression test passed. That observer failure did not stop the GPU campaign.\n'
    (ROOT/'extension_evidence_summary_20261102.md').write_text(prose_blocks(text))
    return text


if __name__=='__main__':section()
