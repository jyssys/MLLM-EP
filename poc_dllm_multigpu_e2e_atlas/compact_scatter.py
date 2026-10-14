"""Exact out-of-place logical KV copy. Never mutate the original cache."""
import torch


def compact_scatter(tensor,source,dim,start,end,step=1):
    result=tensor.clone(memory_format=torch.contiguous_format)
    index=[slice(None)]*tensor.ndim;index[dim]=slice(start,end,step)
    result[tuple(index)].copy_(source)
    return result


class CompactKV:
    def __init__(self,model):self.model=model;self.patches=[];self.active=None;self.records=[]

    def __enter__(self):
        for layer in self.model.model.layers:
            attention=layer.attention;original=attention.forward;lid=layer.layer_id
            def forward(*args,_fn=original,_lid=lid,**kwargs):
                self.active=_lid
                try:return _fn(*args,**kwargs)
                finally:self.active=None
            self.patches.append((attention,'forward',original));attention.forward=forward
        original=torch.Tensor.slice_scatter
        def scatter(tensor,source,*args,_fn=original,**kwargs):
            dim=kwargs.get('dim',args[0] if len(args)>0 else 0)
            start=kwargs.get('start',args[1] if len(args)>1 else None)
            end=kwargs.get('end',args[2] if len(args)>2 else None)
            step=kwargs.get('step',args[3] if len(args)>3 else 1)
            if self.active is None:return _fn(tensor,source,*args,**kwargs)
            assert tensor.ndim==4 and dim==2 and step==1
            assert source.shape[:2]==tensor.shape[:2] and source.shape[3]==tensor.shape[3]
            assert start==tensor.shape[2]-source.shape[2] and end==tensor.shape[2]
            result=compact_scatter(tensor,source,dim,start,end,step)
            self.records.append(dict(layer=self.active,logical_bytes=tensor.numel()*tensor.element_size(),
                source_storage_bytes=tensor.untyped_storage().nbytes(),output_storage_bytes=result.untyped_storage().nbytes(),
                source_rows=source.shape[2],dtype=str(tensor.dtype)))
            return result
        self.patches.append((torch.Tensor,'slice_scatter',original));torch.Tensor.slice_scatter=scatter
        return self

    def __exit__(self,*unused):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)
