"""Late, seed-selected boundary snapshots. No clean timing claim is allowed."""
import os


def install():
    if 'B_NUMERICAL_CAPTURE_SEED' not in os.environ:return
    import torch
    from multiprocessing.util import Finalize
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.worker.executor import ModelRuntime
    seed=int(os.environ['B_NUMERICAL_CAPTURE_SEED'])
    step_limit=int(os.environ.get('B_NUMERICAL_CAPTURE_STEP','4'))
    execute=ModelRunner._execute_batch
    forward=ModelRuntime.forward
    records=[]
    models=[]
    def execute_with_context(self,batch,prepared):
        self.runtime._diagnostic_context=batch
        try:return execute(self,batch,prepared)
        finally:self.runtime._diagnostic_context=None
    ModelRunner._execute_batch=execute_with_context
    def capture_forward(self,batch):
        context=getattr(self,'_diagnostic_context',None)
        selected=[];offset=0
        if context is not None:
            for span in context.spans:
                last=span.token_start+span.token_count-1
                step=last-len(span.request.prompt_token_ids)+1
                if span.request.sampling_params.seed==seed and 0<=step<=step_limit:
                    selected.append((offset+span.token_count-1,step,span.depth,span.request.request_id))
                offset+=span.token_count
        inputs=[batch.hidden[row].detach().clone() if batch.hidden is not None else None for row,_,_,_ in selected]
        result=forward(self,batch)
        for (row,step,depth,rid),hidden in zip(selected,inputs):
            records.append(dict(step=step,depth=depth,operation=batch.operation.value,request_id=rid,
                                rows=len(result.hidden) if result.hidden is not None else None,row=row,
                                input_hidden=hidden,
                                output_hidden=result.hidden[row].detach().clone() if result.hidden is not None else None,
                                logits=result.logits[row].detach().clone() if result.logits is not None else None))
            if result.logits is not None and not models:models.append(self.model)
        return result
    ModelRuntime.forward=capture_forward
    def finish():
        if not records:return
        from .common import ROOT,write
        from .linear_probe import metrics
        folder=ROOT/'raw'/os.environ['B_NUMERICAL_RUN']
        torch.cuda.synchronize()
        geometry=[]
        if models and hasattr(models[0],'model'):
            model=models[0]
            for record in records:
                if record['logits'] is None or record['step']!=step_limit:continue
                x=record['input_hidden']
                reference_norm=None;reference_logits=None
                for count in range(1,9):
                    rows=x.new_zeros((count,len(x)));rows[0].copy_(x)
                    normalized=model.model.norm(rows)
                    logits=model.lm_head(normalized)[0]
                    if reference_norm is None:
                        reference_norm=normalized[0].clone();reference_logits=logits.clone()
                    scores,ids=logits.float().topk(2)
                    geometry.append(dict(step=record['step'],rows=count,norm_to_M1=metrics(reference_norm,normalized[0]),
                                         logits_to_M1=metrics(reference_logits,logits),greedy_argmax=int(logits.argmax()),
                                         top2_ids=ids.tolist(),margin=float(scores[0]-scores[1]),
                                         actual_coda_rows=record['rows'],actual_coda_argmax=int(record['logits'].argmax())))
        for record in records:
            for key in ('input_hidden','output_hidden','logits'):
                if record[key] is not None:record[key]=record[key].cpu()
        torch.save(dict(seed=seed,records=records,scope='Late instrumented actual PD boundary operands. Not clean serving timing.'),
                   folder/f'boundary_operands_{os.getpid()}.pt')
        write(folder/f'actual_coda_geometry_{os.getpid()}.json',geometry)
    Finalize(None,finish,exitpriority=16)
