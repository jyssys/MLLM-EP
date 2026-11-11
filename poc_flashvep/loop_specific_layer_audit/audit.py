"""Separate architecture accounting from measured Serving improvements."""
import csv,hashlib,json,math,struct,time
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_loop_specific_vs_dense_ar_layer_split_analysis_20261112'
PREVIOUS=REPO/'poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111'
CROSS=REPO/'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching'
HUB=Path('/home/esjung/.cache/huggingface/hub')
PRIVATE=Path('/home/esjung/external/looped-pd-abc-rlt')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def checkpoint(model,repository,revision,R,geometry):
    folder=HUB/repository/'snapshots'/revision;headers={};files=[]
    for p in sorted(folder.glob('*.safetensors')):
        with p.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size);value=json.loads(raw)
        value.pop('__metadata__',None);assert not (headers.keys()&value.keys());headers.update(value)
        files.append(dict(file=p.name,header_bytes=size,header_SHA256=hashlib.sha256(raw).hexdigest(),
            physical_file_bytes=p.stat().st_size))
    assert headers
    groups={};parameters={};excluded={}
    for k,row in headers.items():
        n=math.prod(row['shape'])
        if model=='huginn' and k in ('freqs_cis','lm_head.weight'):
            excluded[k]=dict(elements=n,reason='Non-parameter frequency buffer' if k=='freqs_cis' else 'Runtime alias of transformer.wte.weight')
            continue
        recurrent=(k.startswith('model.layers.') or k=='model.norm.weight') if model=='ouro' else k.startswith(('transformer.core_block.','transformer.adapter.'))
        kind='shared_recurrent' if recurrent else 'non_recurrent'
        groups[kind]=groups.get(kind,0)+n;parameters[k]=dict(shape=row['shape'],elements=n,checkpoint_dtype=row['dtype'],group=kind)
    unique=sum(groups.values());runtime=json.loads(geometry.read_text())
    assert unique==runtime['unique_parameters']
    assert 2*unique==runtime['weight_bytes']
    virtual=groups['non_recurrent']+R*groups['shared_recurrent']
    return dict(model=model,revision=revision,fixed_R=R,checkpoint_headers=files,
        checkpoint_config_SHA256=sha(folder/'config.json'),parameters=parameters,excluded=excluded,
        parameter_groups=groups,unique_parameters=unique,runtime_BF16_weight_bytes=2*unique,
        runtime_geometry_source=str(geometry.relative_to(REPO)),runtime_geometry_SHA256=sha(geometry),
        no_alias_unrolled_parameter_elements=virtual,no_alias_unrolled_BF16_bytes=2*virtual,
        status='STRUCTURAL_ACCOUNTING_MATCHED_TO_PREVIOUS_GPU_PARAMETER_INVENTORY',
        scope='Headers supply shapes only. No model tensor payload is loaded. Runtime BF16 bytes are checked against a prior loaded-model inventory. Hypothetical repeated-core copies are not a trained untied AR baseline. An equivalent tied graph can alias those copies and needs no extra parameter storage. Buffers, KV, workspaces, and graph pools are outside parameter bytes.')


def attention(N,R,m,S):
    assert m*N==R*S
    loop=m*N*(N+1)//2
    token=R*(S*(N-S)+S*(S+1)//2)
    return dict(prompt_tokens=N,R=R,D_loops=m,token_suffix=S,equal_recurrent_rows=m*N,
        loop_causal_query_key_pairs_per_core_attention=loop,
        token_causal_query_key_pairs_per_core_attention=token,token_over_loop_pairs=token/loop,
        status='STRUCTURAL_ACCOUNTING_NOT_MEASURED_FLOPS',
        scope='One same-sized core Attention layer, causal pairs only. No Prelude, Coda, padding, launch, or transport cost. The same depth-cut property also applies to an equivalent untied layer stack.')


def mm(x,w):
    return [[sum(a*b for a,b in zip(row,col)) for col in zip(*w)] for row in x]


def cpu_checks():
    W=[[2,1],[-1,3]];H=[[1,2],[3,4]];R=4
    def advance(h,n):
        for _ in range(n):h=mm(h,W)
        return h
    expected=advance(H,R)
    cuts=[]
    for c in range(1,R):
        state=advance(H,c);observed=advance(state,R-c);assert observed==expected
        cuts.append(dict(cut=c,P_calls=c,D_calls=R-c,result_equal=True))
    rows_a=[[1,2]];rows_b=[[3,4]]
    assert mm(rows_a+rows_b,W)==mm(rows_a,W)+mm(rows_b,W)
    U=[[4,-2],[1,5]]
    assert mm(rows_a+rows_b,W)!=mm(rows_a,W)+mm(rows_b,U)
    return dict(passed=True,integer_recurrence_cut_checks=cuts,
        shared_weight_concat_identity=True,distinct_weight_single_GEMM_counterexample=True,
        status='CPU_ALGEBRA_ONLY_NOT_MODEL_CORRECTNESS',
        scope='These checks illustrate state continuation and shared-weight row concatenation. They do not qualify BF16 regrouping, per-depth Attention, KV ownership, GPU timing, or a new Serving implementation.')


def main():
    assert json.loads((PREVIOUS/'state/final_remote_verification.json').read_text())['verified']
    assert not (ROOT/'audit.json').exists(),'Use a new audit namespace'
    ROOT.mkdir(parents=True,exist_ok=True)
    models=[checkpoint('ouro','models--ByteDance--Ouro-1.4B','574fa66cb8bf5abdc979642d01cf2b79b16bfab1',4,
        PREVIOUS/'raw/matched_graph_capacity_ouro_L8192_O16_B124_start1/geometry.json'),
        checkpoint('huginn','models--tomg-group-umd--huginn-0125','bb6621b65e90b6a4b9b29ef88dc83866d450470c',32,
        PREVIOUS/'raw/matched_graph_reserve_capacity_huginn_L3072_B12_start1/geometry.json')]
    sources=[PRIVATE/'vllm_rlt/models/ouro.py',PRIVATE/'vllm_rlt/models/huginn.py',
        PRIVATE/'vllm_rlt/core/scheduler.py',PRIVATE/'vllm_rlt/core/scheduling_policy.py',
        REPO/'poc_flashvep/ref_loopserve/source/vllm/v1/worker/ouro_runner/loop_spec.py',
        REPO/'poc_flashvep/d_local_cross_depth_batching/micro.py',
        REPO/'poc_flashvep/d_local_cross_depth_batching/micro_analysis.py',
        CROSS/'phase_c_gate.json',CROSS/'dense_micro.csv',CROSS/'matched_core_micro.csv']
    evidence=[dict(path=str(p),SHA256=sha(p)) for p in sources]
    quick=REPO/'poc_flashvep/reports/ouro_huginn_loop_token_suffix_128_256_512_quick_comparison.md'
    assert sha(quick)=='729edc4602ad017cdfc4a6e2f427c6d6c309bc95ec00d6dc418df0e36cd2802f'
    result=dict(time=time.time(),GPU_work='NOT_RUN: CPU-only source and header audit',models=models,
        attention_accounting=[attention(4096,4,1,1024),*[attention(2048,32,m,64*m) for m in (1,2,4,8)]],
        CPU_checks=cpu_checks(),source_inventory=evidence,
        predecessor_phase_C=json.loads((CROSS/'phase_c_gate.json').read_text()),
        quick_report_SHA256=sha(quick),previous_discovery_remote_SHA='0efa68913f038bc2b9cab501d5c1b6347d130bbf',
        verdict='NO_CONFIRMED_LOOP_SPLIT_EXCLUSIVE_MECHANISM',
        scope='The audit distinguishes generic depth partitioning, recurrent weight-sharing architecture, and a new PD contribution. It does not execute a Dense AR model or establish new GPU speedup.')
    write(ROOT/'audit.json',result)
    write(ROOT/'configuration.json',dict(time=time.time(),models=[dict(model=x['model'],revision=x['revision'],R=x['fixed_R']) for x in models],
        preceding_campaign_must_be_complete=True,source_SHA256=sha(Path(__file__)),GPU_mapping='NONE',
        math='Original fixed-depth semantics, read-only audit',new_Scheduler='NOT_IMPLEMENTED',new_GPU_measurements='NOT_RUN'))
    print(json.dumps([dict(model=x['model'],parameters=x['unique_parameters'],core=x['parameter_groups']['shared_recurrent'],
        runtime_GiB=x['runtime_BF16_weight_bytes']/2**30,hypothetical_no_alias_GiB=x['no_alias_unrolled_BF16_bytes']/2**30) for x in models]))


if __name__=='__main__':main()
