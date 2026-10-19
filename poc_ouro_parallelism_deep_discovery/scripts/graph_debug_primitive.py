"""CUDA graph API regression. Constant tensors are not model evidence."""
import os
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
configure_native_imports()


def main():
    import torch
    output=Path(os.environ['LOOPED_KV_OUTPUT']);mask=validate_mask();assert len(mask)==1
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[0])]
    value=torch.ones(2048,device='cuda')
    records=[]
    for mode in ['ordinary_explicit_keep','patched_default_keep']:
        if mode=='patched_default_keep':
            import poc_ouro_parallelism_deep_discovery.graph_debug_probe
            graph=torch.cuda.CUDAGraph()
        else:graph=torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph):result=value+1
        graph.instantiate();graph.replay();torch.cuda.synchronize()
        pointer=graph.raw_cuda_graph();assert pointer
        path=output/(mode+'.dot');graph.debug_dump(str(path))
        assert path.exists() and path.stat().st_size>0
        assert torch.equal(result,torch.full_like(result,2))
        records.append(dict(mode=mode,raw_graph_present=True,DOT_created=True,bytes=path.stat().st_size,
                            constant_tensor_output_equal=True))
    write(output/'status.json',dict(success=True,records=records,physical_gpu=mask[0],
          scope='CUDA graph API regression with constant tensors. Not a pretrained model, serving timing, quality, or architecture result.'))


if __name__=='__main__':main()
