"""Keep the full original model and default compiler for a stronger token arm.

Only producer token ownership and the remote prefix count change. The adapter
does not replace a model forward or a Transformer layer. Post-run observations
read the original runner's query geometry and completed hidden state.
"""
import time
from .adapter import Adapter


class StockTokenAdapter(Adapter):
    def install(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
        assert self.cfg['axis_kind']=='Token' and self.cfg.get('token_stock_model')
        me=self;execute=GPUModelRunner.execute_model
        def executed(runner,*a,**kw):
            result=execute(runner,*a,**kw)
            scheduled=a[0] if a else kw.get('scheduler_output')
            dummy=kw.get('dummy_run',a[2] if len(a)>2 else False)
            if dummy or scheduled is None or scheduled.total_num_scheduled_tokens==0:return result
            frame=runner.execute_model_state
            if frame is not None and frame.hidden_states is not None:
                batch=frame.input_batch
                if all(me.external(rid) in me.specs for rid in batch.req_ids):
                    assert all(not me.partition(rid).upper for rid in batch.req_ids)
                    me.completed_full_graph(runner,execution='UNMODIFIED_FULL36_STANDARD_RUNNER')
            return result
        GPUModelRunner.execute_model=executed
        attention=GPUModelRunner.prepare_attn
        def prepared_attention(runner,batch,*a,**kw):
            result=attention(runner,batch,*a,**kw)
            if me.role=='decode' and me.cfg.get('axis_state_capture') and not me.cfg.get('axis_import_hash_after_model'):
                for i,rid in enumerate(batch.req_ids):
                    if me.external(rid) not in me.specs:continue
                    part=me.partition(rid)
                    me.capture_state(runner,batch,i,part.prefix,36,None,'import_'+me.external(rid))
            return result
        GPUModelRunner.prepare_attn=prepared_attention
        if self.role=='decode':
            original=NixlPullConnectorScheduler.get_num_new_matched_tokens
            def matched(connector,request,computed):
                part=me.partition(request.request_id)
                if request.kv_transfer_params and request.kv_transfer_params.get('do_remote_prefill'):
                    assert request.kv_transfer_params['remote_num_tokens']==part.prefix
                    value=max(0,part.prefix-computed)
                    return value,value>0
                return original(connector,request,computed)
            NixlPullConnectorScheduler.get_num_new_matched_tokens=matched
        self.rows.append(dict(event='STOCK_MODEL_TOKEN_PATH',model_forward_replaced=False,
            decoder_layer_forward_replaced=False,original_full36_model=True,time=time.monotonic(),
            scope='Full36 coverage follows the unchanged standard model graph for each observed runner query. '
                'This observation does not assign exclusive kernel times. Strict teacher-state tests remain required.'))
