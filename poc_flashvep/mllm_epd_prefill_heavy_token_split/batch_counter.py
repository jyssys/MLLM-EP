"""Count actual prepared query geometry without CUDA events or tensor copies."""
from collections import Counter
import os
import re
import time

from .common import ROOT,write

QUERY_FIELDS=['requests','prompt_only_requests','decode_only_requests','mixed_requests','query_rows',
    'prompt_query_rows','decode_query_rows']
PADDING_FIELDS=['requests','decode_requests','query_rows','requests_after_padding',
    'query_rows_after_padding','graph_mode']
SCHEDULE_FIELDS=['requests','generation_phase_requests','query_rows']


def query_geometry(batch,scheduled):
    count=int(batch.num_reqs)
    return geometry(count,scheduled,batch.num_computed_tokens_cpu[:count],batch.num_prompt_tokens[:count])


def v2_query_geometry(batch):
    assert batch.num_draft_tokens==0,'Speculative query geometry is outside this counter contract'
    value=geometry(int(batch.num_reqs),batch.num_scheduled_tokens,
        batch.num_computed_prefill_tokens_np,batch.prefill_len_np)
    assert value[4]==batch.num_tokens
    return value


def geometry(count,scheduled,computed_counts,prompt_counts):
    assert len(scheduled)==count and count>0
    assert len(computed_counts)==len(prompt_counts)==count
    prompt_only=decode_only=mixed=prompt_rows=decode_rows=0
    for number,computed,prompt in zip(scheduled,computed_counts,prompt_counts):
        number,computed,prompt=int(number),int(computed),int(prompt)
        assert number>0 and computed>=0 and prompt>0
        remaining=max(0,prompt-computed);prefill=min(number,remaining);decode=number-prefill
        prompt_rows+=prefill;decode_rows+=decode
        if prefill and decode:mixed+=1
        elif prefill:prompt_only+=1
        else:decode_only+=1
    assert prompt_only+decode_only+mixed==count
    return (count,prompt_only,decode_only,mixed,prompt_rows+decode_rows,prompt_rows,decode_rows)


class BatchCounter:
    def __init__(self,runner,name,decode_only=False):
        assert re.fullmatch('[A-Za-z0-9_.-]+',name)
        assert isinstance(decode_only,bool)
        self.decode_only=decode_only
        if decode_only:
            assert os.environ['MLLM_EPD_ROLE']=='decode' and not hasattr(runner,'_prepare_inputs'), (
                'Decode-only padding evidence requires the V2 decode worker')
        self.runner=runner;self.patches=[];self.active=False
        self.folder=ROOT/'raw/batch_counter'/name/(os.environ['MLLM_EPD_ROLE']+'_gpu'+os.environ['CUDA_VISIBLE_DEVICES'])
        assert not self.folder.exists();self.folder.mkdir(parents=True)
        self.prepared=Counter();self.executed=Counter();self.encoded=Counter();self.errors=Counter()
        self.padded=Counter();self.generation_schedule=Counter()

    def patch(self,key,wrapped,owner=None):
        owner=self.runner if owner is None else owner
        self.patches.append((owner,key,key in vars(owner),vars(owner).get(key),wrapped))
        setattr(owner,key,wrapped)

    def begin(self):
        assert not self.active
        execute=self.runner.execute_model;me=self
        legacy=hasattr(self.runner,'_prepare_inputs')
        prepare_key='_prepare_inputs' if legacy else 'prepare_inputs'
        prepare=getattr(self.runner,prepare_key)
        if not self.decode_only:
            encoder_owner=self.runner if legacy else self.runner.model_state.encoder_runner
            encoder_key='_execute_mm_encoder' if legacy else 'execute_mm_encoder'
            encode=getattr(encoder_owner,encoder_key)
        def prepared(output,scheduled,*args,**kwargs):
            try:
                geometry=query_geometry(me.runner.input_batch,scheduled)
                assert geometry[4]==output.total_num_scheduled_tokens
            except Exception as error:
                geometry=None;me.errors['prepare:'+type(error).__name__+': '+str(error)]+=1
            result=prepare(output,scheduled,*args,**kwargs)
            if geometry is not None:me.prepared[geometry]+=1
            return result
        def prepared_v2(output,*args,**kwargs):
            result=prepare(output,*args,**kwargs)
            try:
                actual=v2_query_geometry(result)
                assert actual[4]==output.total_num_scheduled_tokens
                if me.decode_only:
                    assert actual[0]==len(output.num_scheduled_tokens)
                    assert all(output.num_scheduled_tokens[request_id]==int(rows)
                        for request_id,rows in zip(result.req_ids,result.num_scheduled_tokens,strict=True))
                    descriptor=args[0] if args else kwargs['batch_desc']
                    me.padded[(actual[0],actual[2]+actual[3],actual[4],
                        int(result.num_reqs_after_padding),int(result.num_tokens_after_padding),
                        descriptor.cg_mode.name)]+=1
                me.prepared[actual]+=1
            except Exception as error:me.errors['prepare_v2:'+type(error).__name__+': '+str(error)]+=1
            return result
        def executed(output,*args,**kwargs):
            result=execute(output,*args,**kwargs)
            key=(len(output.num_scheduled_tokens),int(output.total_num_scheduled_tokens),
                sum(len(value) for value in output.scheduled_encoder_inputs.values()))
            me.executed[key]+=1
            if me.decode_only and key[0]:
                try:
                    cached=output.scheduled_cached_reqs
                    generation_count=sum(req_id in output.num_scheduled_tokens and output_count>0
                        for req_id,output_count in zip(cached.req_ids,cached.num_output_tokens,strict=True))
                    me.generation_schedule[(key[0],generation_count,key[1])]+=1
                except Exception as error:me.errors['schedule:'+type(error).__name__+': '+str(error)]+=1
            return result
        def encoded(output,*args,**kwargs):
            result=encode(output,*args,**kwargs)
            try:
                assert isinstance(result,(list,tuple))
                # Shape lookup reads host Tensor metadata. No item(), CPU copy,
                # event, stream change, GPU allocation or synchronization.
                me.encoded[(len(result),sum(int(tensor.shape[0]) for tensor in result))]+=1
            except Exception as error:me.errors['encoder:'+type(error).__name__+': '+str(error)]+=1
            return result
        self.patch('execute_model',executed);self.patch(prepare_key,prepared if legacy else prepared_v2)
        if not self.decode_only:
            self.patch(encoder_key,encoded,encoder_owner)
        self.runner_API='V1_PRIVATE_PREPARE' if legacy else 'V2_RETURNED_INPUT_BATCH'
        self.active=True;self.start=time.monotonic()
        return dict(folder=str(self.folder),observer='Counter only. No CUDA events, copies, synchronization or per-step timestamps.',
            status='DIAGNOSTIC_PENDING_CPC',counter_model_input_change=False,runner_API=self.runner_API,
            decode_only=self.decode_only)

    def close(self):
        assert self.active
        for owner,key,existed,original,wrapped in reversed(self.patches):
            assert getattr(owner,key) is wrapped, 'A different observer replaced an active counter hook'
            if existed:setattr(owner,key,original)
            else:delattr(owner,key)
        self.active=False
        prepared=[dict(zip(QUERY_FIELDS,key),calls=value) for key,value in sorted(self.prepared.items())]
        executed=[dict(requests=key[0],scheduled_query_rows=key[1],scheduled_encoder_items=key[2],calls=value)
            for key,value in sorted(self.executed.items())]
        encoded=[dict(output_images=key[0],output_visual_rows=key[1],calls=value) for key,value in sorted(self.encoded.items())]
        result=dict(begin_monotonic=self.start,end_monotonic=time.monotonic(),pid=os.getpid(),
            prepared_query_histogram=prepared,worker_schedule_histogram=executed,encoder_output_histogram=encoded,
            errors=dict(self.errors),original_methods_restored=True,runner_API=self.runner_API,
            valid=bool(executed) and not self.errors,
            scope='Actual logical query rows at input preparation and encoder output shapes. '
                'Prompt rows are not named decode solely because Scheduler has already advanced computed-token counters. '
                'Empty worker cleanup calls are separate. CUDA-graph padded kernel shapes, exclusive GPU service, '
                'timeline, queue and safe windows are not measured. Observer CPC qualification and model quality remain separate.')
        if self.decode_only:
            result.update(
                scheduler_generation_phase_histogram=[dict(zip(SCHEDULE_FIELDS,key),calls=count)
                    for key,count in sorted(self.generation_schedule.items())],
                prepared_decode_padding_histogram=[dict(zip(PADDING_FIELDS,key),calls=count)
                    for key,count in sorted(self.padded.items())],
                scope='Prepared logical decode queries and vLLM CUDA graph dispatch padding. '
                    'Scheduler generation phase counts scheduled cached requests with prior output. '
                    'This phase count and prompt-versus-decode query classification are reported separately. '
                    'No physical kernel shape or GPU timeline was measured.')
            scheduled_calls=sum(value for key,value in self.executed.items() if key[0])
            result['valid']=result['valid'] and scheduled_calls>0 and (
                scheduled_calls==sum(self.prepared.values())==sum(self.generation_schedule.values())==sum(self.padded.values()))
            result['decode_maxima']=dict(
                scheduled_requests=max((key[0] for key in self.generation_schedule),default=0),
                scheduler_generation_phase_requests=max((key[1] for key in self.generation_schedule),default=0),
                prepared_decode_requests=max((key[1] for key in self.padded),default=0),
                prepared_query_rows=max((key[2] for key in self.padded),default=0),
                prepared_requests_after_padding=max((key[3] for key in self.padded),default=0),
                prepared_query_rows_after_padding=max((key[4] for key in self.padded),default=0))
        write(self.folder/'histograms.json',result)
        receipt=dict(folder=str(self.folder),valid=result['valid'],errors=result['errors'],
            prepared_calls=sum(self.prepared.values()),worker_calls=sum(self.executed.values()),
            encoder_calls=sum(self.encoded.values()),original_methods_restored=True)
        if self.decode_only:
            receipt['decode_maxima']=result['decode_maxima']
        return receipt
