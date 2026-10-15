import unittest
import importlib.util
import types
from unittest.mock import patch
from poc_looped_multigpu_kv_placement.contract import GPUS,MASK,ROOT,sha
from poc_looped_multigpu_kv_placement.analysis import bootstrap,kv_bytes_per_token


class ContractTests(unittest.TestCase):
    def test_device_scope(self):
        self.assertEqual(GPUS,[4,5,6,7]);self.assertEqual(MASK,'4,5,6,7')

    def test_ouro_actual_depth(self):
        self.assertEqual(kv_bytes_per_token(48*4,16,128),1572864)
        self.assertEqual(kv_bytes_per_token(48*4,16,128,tp=4),393216)

    def test_huginn_divisibility(self):
        self.assertEqual(kv_bytes_per_token(2+4*16+2,55,96),1436160)
        for tp in [2,4]:
            with self.assertRaises(AssertionError):kv_bytes_per_token(68,55,96,tp=tp)

    def test_bootstrap_request_units(self):
        result=bootstrap([1,1,1]);self.assertEqual(result['CI95'],[1,1]);self.assertEqual(result['N'],3)

    def test_frozen_token_identity(self):
        import json
        manifest=json.loads((ROOT/'results/frozen_workload.json').read_text())
        self.assertEqual(sha(ROOT/'results/frozen_tokens.json.gz'),manifest['token_artifact_sha256'])

    def test_filled_positions_are_not_output_count(self):
        # The last returned token has no subsequent forward.
        self.assertEqual((32704+64-1)*kv_bytes_per_token(192,16,128),51538034688)

    def test_shared_module_is_observed_once(self):
        class Event:
            def record(self):pass
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(Event=lambda **unused:Event()))
        spec=importlib.util.spec_from_file_location('observer_cpu_test',ROOT/'loop_observer.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'torch':fake}):spec.loader.exec_module(module)
        observer=module.LoopObserver(None);calls=[]
        obj=types.SimpleNamespace(forward=lambda:calls.append(1))
        observer.wrap(obj,'forward','ROPE');observer.wrap(obj,'forward','ROPE')
        observer.active=(0,0,1);obj.forward()
        self.assertEqual(len(observer.records),1);self.assertEqual(calls,[1])


if __name__=='__main__':unittest.main()
