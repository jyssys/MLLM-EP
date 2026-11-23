"""Observe original model-call boundaries without changing its arguments."""
import atexit
import hashlib
import inspect
import textwrap
import time

REPLACEMENTS=(
    ('model_output = self.cudagraph_manager.run_fullgraph(batch_desc)',
     'model_output = _pc_forward_call(self, "FULL", lambda: self.cudagraph_manager.run_fullgraph(batch_desc), input_batch)'),
    ('model_output = self.cudagraph_manager.run_pw_graph(\n                    self.model, model_inputs\n                )',
     'model_output = _pc_forward_call(self, "PIECEWISE", lambda: self.cudagraph_manager.run_pw_graph(self.model, model_inputs), input_batch)'),
    ('model_output = self.model(**model_inputs)',
     'model_output = _pc_forward_call(self, "NONE", lambda: self.model(**model_inputs), input_batch)'),
)


def rewrite(source):
    value=source
    for before,after in REPLACEMENTS:
        assert value.count(before)==1,('Installed model-runner seam changed',before)
        value=value.replace(before,after)
    restored=value
    for before,after in REPLACEMENTS:restored=restored.replace(after,before)
    assert restored==source,'Only the three original call boundaries may change'
    return value


def execution_namespace(original):
    # Later adapter installation changes dispatch_cg_and_sync_dp in this module.
    # A copied dict would freeze the pre-guard dispatcher and change execution.
    return inspect.unwrap(original).__globals__


class ModelTrace:
    def __init__(self,cfg,role,folder,write):
        self.cfg,self.role,self.folder,self.write=cfg,role,folder,write
        self.active=False;self.origin=None;self.rows=[];self.events=[];self.closed=False

    def boundary(self):
        import torch
        # The coordinator has not started its arrival clock. Both warm controls ended.
        self.origin=torch.cuda.Event(enable_timing=True)
        left=time.monotonic_ns();self.origin.record();self.origin.synchronize();right=time.monotonic_ns()
        self.anchor=dict(host_lower_ns=left,host_upper_ns=right,uncertainty_ns=right-left,
            scope='The origin CUDA event executed inside these host bounds before measured arrivals. '
                'Later GPU event timestamps use origin.elapsed_time. They are enclosures,not exclusive kernel time.')
        self.rows.clear();self.events.clear();self.active=True

    def forward(self,runner,mode,callback,batch):
        import torch
        if not self.active or torch.cuda.is_current_stream_capturing():return callback()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        row=dict(host_enter=time.monotonic(),mode=mode,valid_queries=int(batch.num_tokens),
            padded_queries=int(batch.num_tokens_after_padding),actual_requests=int(batch.num_reqs),
            queries=[dict(request_id=rid.split('-')[0],start=int(batch.num_computed_tokens_np[i]),
                count=int(batch.num_scheduled_tokens[i])) for i,rid in enumerate(batch.req_ids)])
        begin.record()
        result=callback()
        end.record();row['host_exit']=time.monotonic()
        self.rows.append(row);self.events.append((begin,end))
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        if self.origin is not None:
            import torch
            torch.cuda.synchronize()
            for row,(begin,end) in zip(self.rows,self.events):
                row.update(GPU_start_ms_from_origin=self.origin.elapsed_time(begin),
                    GPU_end_ms_from_origin=self.origin.elapsed_time(end),GPU_enclosure_ms=begin.elapsed_time(end))
                assert row['GPU_enclosure_ms']>=0
        self.write(self.folder/f'P_model_forward_events_{self.role}.json',dict(origin=getattr(self,'anchor',None),rows=self.rows,
            evidence='GPU_EVENT_ENCLOSURE',scope='Actual original FULL/PIECEWISE/NONE model-call boundaries. '
                'No model argument,output,sampling or client timestamp changed. No per-step device synchronization. '
                'The final synchronization occurs after the measured client trace. '
                'GPU enclosures include stream dependencies,host launch gaps and kernels. Use CUPTI union for exclusive kernel activity.'))


def install(cfg,role,folder,info):
    if role!='prefill' or not cfg.get('P_event_timing'):return None
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from poc_flashvep.dense_ar_longbench400_pd.common import write
    probe=ModelTrace(cfg,role,folder,write);original=GPUModelRunner.execute_model
    source=textwrap.dedent(inspect.getsource(original));modified=rewrite(source)
    # Keep the live model-runner namespace. Subsequent legal graph guards and
    # teacher dispatch changes must remain visible to the observed method.
    unwrapped=inspect.unwrap(original)
    environment=execution_namespace(original)
    assert '_pc_forward_call' not in environment,'Install one P observer per worker'
    environment['_pc_forward_call']=probe.forward
    generated=folder/'observed_model_runner_execute_model.py';generated.write_text(modified)
    sentinel=object();prior=environment.get(unwrapped.__name__,sentinel)
    exec(compile(modified,str(generated),'exec'),environment)
    GPUModelRunner.execute_model=environment[unwrapped.__name__]
    if prior is sentinel:del environment[unwrapped.__name__]
    else:environment[unwrapped.__name__]=prior
    write(folder/'model_call_observer_provenance.json',dict(original_source_SHA256=hashlib.sha256(source.encode()).hexdigest(),
        observed_source_SHA256=hashlib.sha256(modified.encode()).hexdigest(),exactly_three_call_boundaries=True,
        argument_and_output_passthrough=True,installed_module=unwrapped.__module__,
        live_module_namespace=True,later_graph_guard_visible=True,
        scope='Process-local method clone. The installed vLLM file remains unchanged. '
            'The source transformation is reversible byte-for-byte. A separate CPC observer gate is mandatory.'))
    atexit.register(probe.close);return probe
