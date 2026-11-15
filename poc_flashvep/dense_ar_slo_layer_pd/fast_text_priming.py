"""Prime an exact ByteLevel stream after a verified ASCII context boundary."""
import hashlib,json,time
from .common import *


def context_suffix(ids,suffix):
    if len(ids)>=len(suffix) and ids[-len(suffix):]==suffix:return list(suffix)
    if len(ids)>len(suffix) and ids[-len(suffix)-1:-1]==suffix:return list(suffix)+ids[-1:]
    return None


def stream(tokenizer,ids,suffix,skip_special_tokens=True):
    from tokenizers.decoders import DecodeStream
    from ..dense_ar_layerwise_pd.output_priming import primed_stream
    selected=context_suffix(ids,suffix) if skip_special_tokens else None
    if selected is None:return primed_stream(tokenizer,ids,skip_special_tokens),False
    value=DecodeStream(ids=selected[:-1],skip_special_tokens=True)
    value.step(tokenizer,selected[-1]);return value,True


def install(event,verify=False):
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer
    contract=json.loads((ROOT/'configs/verified_ByteLevel_text_priming.json').read_text())
    cfg=json.loads(Path(os.environ['DENSE_CONFIG']).read_text())
    contract_SHA256=digest(ROOT/'configs/verified_ByteLevel_text_priming.json')
    if cfg.get('fast_priming_contract_SHA256') is not None:
        assert cfg['fast_priming_contract_SHA256']==contract_SHA256
    assert contract['model_revision']==MODEL_REVISION and digest(MODEL/'tokenizer.json')==contract['tokenizer_SHA256']
    assert contract['decoder']==dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False)
    suffix=contract['generation_suffix_ids']
    original_init=FastIncrementalDetokenizer.__init__;original_decode=FastIncrementalDetokenizer.decode_next
    def initialized(self,tokenizer,request):
        original_init(self,tokenizer,request)
        ids=request.prompt_token_ids
        if not ids:return
        start=time.monotonic();self.stream,trimmed=stream(self.tokenizer,ids,suffix,self.skip_special_tokens)
        self._dense_shadow=None;self._dense_external=request.external_req_id
        if verify:
            shadow=FastIncrementalDetokenizer.__new__(FastIncrementalDetokenizer)
            original_init(shadow,tokenizer,request);self._dense_shadow=shadow
        event('VERIFIED_BYTELEVEL_TEXT_PRIMED',request_id=request.request_id,
            external_request_id=request.external_req_id,prompt_tokens=len(ids),trimmed=trimmed,
            contract_SHA256=contract_SHA256,contract_pinned_in_configuration='fast_priming_contract_SHA256' in cfg,
            host_elapsed_ms=1000*(time.monotonic()-start),
            scope='The verified ASCII generation suffix ends the UTF8 prefix state. Keep a Native bridge token when present. Unknown contexts retain original full priming. All generated token IDs and text processing remain unchanged.')
    def decoded(self,next_token_id):
        text=original_decode(self,next_token_id);shadow=getattr(self,'_dense_shadow',None)
        if shadow is not None:
            expected=original_decode(shadow,next_token_id)
            assert text==expected,'ByteLevel suffix and original complete-prefix text differ'
            event('VERIFIED_BYTELEVEL_TEXT_PARITY',external_request_id=self._dense_external,
                token_id=next_token_id,passed=True,text_SHA256=hashlib.sha256(text.encode()).hexdigest())
        return text
    FastIncrementalDetokenizer.__init__=initialized;FastIncrementalDetokenizer.decode_next=decoded


def cpu_check():
    target=ROOT/'configs/verified_ByteLevel_text_priming.json'
    if target.exists():
        contract=json.loads(target.read_text())
        assert contract['CPU_gate_SHA256']==digest(ROOT/'analysis/verified_ByteLevel_text_CPU_gate.json')
        print('CPU_TEXT_GATE_ALREADY_FROZEN');return
    import random,statistics
    from tokenizers import Tokenizer
    from ..dense_ar_layerwise_pd.output_priming import primed_stream
    from .analyze import records,percentile
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    manifest=json.loads((ROOT/'data/unseen_manifest.json').read_text());suffix=manifest['generation_suffix_ids']
    decoder=json.loads((MODEL/'tokenizer.json').read_text())['decoder']
    assert decoder==dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False)
    assert tokenizer.decode(suffix).isascii() and tokenizer.decode(suffix).endswith('\n\n')
    files=[PREVIOUS/'raw/baseline_Native_O200_N64_q4p8_s21_v1/requests.jsonl',
        PREVIOUS/'raw/heldout_Native_O200_N128_q1.5_s2026111381_v1/requests.jsonl']
    cases=[];cost_before=[];cost_after=[]
    for path in files:
        for row in records(path):
            for bridge in (False,True):
                ids=row['input_ids']+row['output_ids'][:1] if bridge else row['input_ids']
                outputs=row['output_ids'][1:] if bridge else row['output_ids']
                start=time.perf_counter();a=primed_stream(tokenizer,ids);cost_before.append(1000*(time.perf_counter()-start))
                start=time.perf_counter();b,trimmed=stream(tokenizer,ids,suffix);cost_after.append(1000*(time.perf_counter()-start))
                expected=[a.step(tokenizer,t) for t in outputs];actual=[b.step(tokenizer,t) for t in outputs]
                assert expected==actual,(row['sample_id'],bridge)
                cases.append(dict(prompt_id=row['prompt_id'],Native_bridge=bridge,trimmed=trimmed,
                    output_tokens=len(outputs),passed=True))
    rng=random.Random(1414);fuzz=[];ids=records(files[0])[0]['input_ids']
    for i in range(1000):
        outputs=[rng.randrange(tokenizer.get_vocab_size()) for _ in range(24)]
        a=primed_stream(tokenizer,ids);b,trimmed=stream(tokenizer,ids,suffix)
        for t in outputs:
            def step(s):
                try:return ('TEXT',s.step(tokenizer,t))
                except Exception as e:return ('ERROR',type(e).__name__,str(e))
            expected=step(a);actual=step(b)
            assert expected==actual,(i,t,expected,actual)
        fuzz.append(dict(case=i,passed=True))
    result=dict(cases=cases,fuzz=fuzz,passed=True,
        priming_wall_ms=dict(original_median=statistics.median(cost_before),
            candidate_median=statistics.median(cost_after),original_p95=percentile(cost_before,.95),
            candidate_p95=percentile(cost_after,.95)),
        evidence='CORRECTNESS_ONLY',scope='CPU streamed-text parity against the original decoder. This is not GPU Serving performance. Verification includes all emitted text, not only token IDs.')
    write(ROOT/'analysis/verified_ByteLevel_text_CPU_gate.json',result)
    contract=dict(model_revision=MODEL_REVISION,tokenizer_SHA256=digest(MODEL/'tokenizer.json'),decoder=decoder,
        generation_suffix_ids=suffix,CPU_gate_SHA256=digest(ROOT/'analysis/verified_ByteLevel_text_CPU_gate.json'),
        scope='Specialized exact stream initialization for the verified Qwen3 ByteLevel decoder and complete ASCII generation suffix. Other contexts use original full initialization.')
    if target.exists():assert json.loads(target.read_text())==contract
    else:write(target,contract)
    print('CPU_TEXT_GATE',len(cases),len(fuzz),result['priming_wall_ms'])


if __name__=='__main__':cpu_check()
