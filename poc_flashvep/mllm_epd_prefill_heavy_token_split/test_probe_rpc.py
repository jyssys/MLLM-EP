"""Exercise the actual endpoint body so the diagnostic option reaches RPC."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock


class ProbeRPCTest(unittest.IsolatedAsyncioTestCase):
    async def test_attention_flag_reaches_worker(self):
        source=Path(__file__).with_name('server_entry.py')
        nodes=[n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n,ast.AsyncFunctionDef) and n.name=='llm_probe_begin']
        node,=nodes
        node.decorator_list=[]
        worker=AsyncMock(return_value=['recorded'])
        app=types.SimpleNamespace(state=types.SimpleNamespace(
            engine_client=types.SimpleNamespace(collective_rpc=worker)))
        namespace=dict(app=app,Request=object)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        request=types.SimpleNamespace(json=AsyncMock(return_value=dict(
            name='actual-diagnostic',max_logits=4,include_decode_inputs=True,include_attention=True)))
        result=await namespace['llm_probe_begin'](request)
        self.assertEqual(result,['recorded'])
        self.assertEqual(worker.call_args.kwargs['args'],('actual-diagnostic',4,True,True,False))

    async def test_gemm_diagnostic_flag_reaches_worker(self):
        source=Path(__file__).with_name('server_entry.py')
        node,=[n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n,ast.AsyncFunctionDef) and n.name=='llm_probe_begin']
        node.decorator_list=[]
        worker=AsyncMock(return_value=['recorded'])
        app=types.SimpleNamespace(state=types.SimpleNamespace(
            engine_client=types.SimpleNamespace(collective_rpc=worker)))
        namespace=dict(app=app,Request=object)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        request=types.SimpleNamespace(json=AsyncMock(return_value=dict(
            name='actual-gemm-diagnostic',max_logits=4,include_decode_inputs=True,
            include_attention=True,include_gemm=True)))
        await namespace['llm_probe_begin'](request)
        self.assertEqual(worker.call_args.kwargs['args'],('actual-gemm-diagnostic',4,True,True,True))


if __name__=='__main__':unittest.main()
