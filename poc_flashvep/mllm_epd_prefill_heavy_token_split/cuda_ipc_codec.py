"""Serialize only the supported BF16 CUDA-IPC reduction fields as JSON."""
import base64


def encode_reduction_args(args):
    import torch
    assert len(args)==15 and args[0] is torch.Tensor and args[4] is torch.storage.TypedStorage
    assert args[5] is torch.bfloat16 and args[10] is False
    value=dict(format='torch_cuda_ipc_bf16_v1',shape=list(args[1]),stride=list(args[2]),
        tensor_offset=args[3],device=args[6],storage_bytes=args[8],storage_offset_bytes=args[9],
        ref_counter_offset=args[12],event_sync_required=args[14])
    for name,index in [('memory_handle',7),('ref_counter_handle',11),('event_handle',13)]:
        assert isinstance(args[index],bytes)
        value[name]=base64.b64encode(args[index]).decode('ascii')
    decode_reduction_args(value)
    return value


def decode_reduction_args(value):
    import torch
    assert value['format']=='torch_cuda_ipc_bf16_v1'
    shape=value['shape'];stride=value['stride']
    assert len(shape)==len(stride)==2 and all(type(x) is int for x in [*shape,*stride])
    assert 0<shape[0]<=262144 and shape[1]==3584 and stride[0]>=shape[1] and stride[1]==1
    for key in ['tensor_offset','device','storage_bytes','storage_offset_bytes','ref_counter_offset']:
        assert type(value[key]) is int and value[key]>=0,key
    assert type(value['event_sync_required']) is bool
    assert (value['tensor_offset']+(shape[0]-1)*stride[0]+shape[1])*2<=value['storage_bytes']
    handles=[]
    for name in ['memory_handle','ref_counter_handle','event_handle']:
        assert isinstance(value[name],str) and len(value[name])<=8192
        handles.append(base64.b64decode(value[name],validate=True))
    # Torch serializes a versioned allocator handle, not just cudaIpcMemHandle_t.
    # Keep that byte string opaque. The native Torch importer checks its ABI.
    assert 0<len(handles[0])<=4096 and 0<len(handles[1])<=256 and len(handles[2])==64
    return (torch.Tensor,torch.Size(shape),tuple(stride),value['tensor_offset'],torch.storage.TypedStorage,
        torch.bfloat16,value['device'],handles[0],value['storage_bytes'],value['storage_offset_bytes'],False,
        handles[1],value['ref_counter_offset'],handles[2],value['event_sync_required'])
