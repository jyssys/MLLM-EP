"""Separate retained-graph diagnostic. Not the primary timing policy."""
import ctypes
import hashlib
from collections import Counter
from pathlib import Path
import torch
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe

# The primary engine destroys cudaGraph_t after instantiation. Retain it only here.
_original_new=torch.cuda.CUDAGraph.__new__
_original_init=torch.cuda.CUDAGraph.__init__
def _retained_new(cls,keep_graph=False):
    return _original_new(cls,keep_graph=True)
def _retained_init(self,keep_graph=False):
    # The C++ constructor also consumes this flag. __new__ alone is insufficient.
    return _original_init(self,keep_graph=True)
torch.cuda.CUDAGraph.__new__=staticmethod(_retained_new)
torch.cuda.CUDAGraph.__init__=_retained_init


class GraphProbe(FourModelProbe):
    def sampler_bypass(self,enabled):
        sampler=self.model_runner.sampler
        if not hasattr(self,'sampler_wrapper'):self.sampler_wrapper=sampler.sample
        sampler.sample=self.original_sampler_sample if enabled else self.sampler_wrapper
        return True

    def graph_nodes(self):
        paths=[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line]
        assert paths,'Use the actually loaded CUDA runtime'
        lib=ctypes.CDLL(paths[0]);ptr=ctypes.c_void_p;size=ctypes.c_size_t
        lib.cudaGraphGetNodes.argtypes=[ptr,ctypes.POINTER(ptr),ctypes.POINTER(size)]
        lib.cudaGraphGetEdges.argtypes=[ptr,ctypes.POINTER(ptr),ctypes.POINTER(ptr),ctypes.POINTER(size)]
        lib.cudaGraphNodeGetType.argtypes=[ptr,ctypes.POINTER(ctypes.c_int)]
        def check(code):assert code==0,('CUDA graph introspection error',code)
        result=[]
        for descriptor,graph in self.model_runner.cudagraph_manager.graphs.items():
            graph.instantiate();raw=ptr(graph.raw_cuda_graph())
            n=size();check(lib.cudaGraphGetNodes(raw,None,ctypes.byref(n)))
            nodes=(ptr*n.value)();check(lib.cudaGraphGetNodes(raw,nodes,ctypes.byref(n)))
            types=[]
            for node in nodes:
                kind=ctypes.c_int();check(lib.cudaGraphNodeGetType(node,ctypes.byref(kind)));types.append(kind.value)
            edges=size();check(lib.cudaGraphGetEdges(raw,None,None,ctypes.byref(edges)))
            sources=(ptr*edges.value)();destinations=(ptr*edges.value)()
            edge_status=lib.cudaGraphGetEdges(raw,sources,destinations,ctypes.byref(edges)) if edges.value else 0
            # Non-default dependency metadata can make this legacy edge API lossy.
            # Node/edge counts remain direct queries. Do not fabricate a topology hash.
            if edge_status:
                check(0 if edge_status==402 else edge_status)
                lib.cudaGetLastError()
                result.append(dict(descriptor=repr(descriptor),node_count=n.value,edge_count=edges.value,
                    node_type_counts=dict(Counter(types)),dependency_type_fingerprint_8_rounds='UNRESOLVED',
                    edge_query_error_code=edge_status,runtime_library=paths[0],
                    scope='Retained graph counts only. Legacy API cannot expose all dependency metadata. Not primary timing.'))
                continue
            indices={int(node):i for i,node in enumerate(nodes)}
            incoming=[[] for _ in types];outgoing=[[] for _ in types]
            for a,b in zip(sources,destinations):
                u=indices[int(a)];v=indices[int(b)];outgoing[u].append(v);incoming[v].append(u)
            labels=[str(t) for t in types]
            for _ in range(8):
                labels=[hashlib.sha256(repr((labels[i],sorted(labels[j] for j in incoming[i]),
                    sorted(labels[j] for j in outgoing[i]))).encode()).hexdigest() for i in range(len(types))]
            fingerprint=hashlib.sha256('\n'.join(sorted(labels)).encode()).hexdigest()
            result.append(dict(descriptor=repr(descriptor),node_count=n.value,edge_count=edges.value,
                node_type_counts=dict(Counter(types)),dependency_type_fingerprint_8_rounds=fingerprint,runtime_library=paths[0],
                scope='Retained cudaGraph_t diagnostic. Not primary keep_graph=False timing.'))
        return result
