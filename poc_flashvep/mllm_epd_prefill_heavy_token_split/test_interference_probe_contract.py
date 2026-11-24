"""The optional decode probe must retain the original Native command and NIXL connector."""
import json
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ProbeRuntimeContractTest(unittest.TestCase):
    def test_launcher_rejects_health_from_a_different_live_proxy(self):
        from . import interference_probe_control as controller
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir();(root/'logs').mkdir()
            (root/'state/server_actual_E4.json').write_text(json.dumps(dict(executed_source_cwd=directory)))
            process=SimpleNamespace(pid=12345,poll=lambda:None)
            class OldHealth(io.BytesIO):status=200
            with patch.object(controller,'ROOT',root),patch.object(controller.subprocess,'Popen',return_value=process),\
                    patch.object(controller.urllib.request,'urlopen',return_value=OldHealth(b'{"name":"unrelated_old_proxy"}')):
                with self.assertRaises(AssertionError):controller.launch_proxy('new_study',['actual_E4'])

    def test_teacher_rpc_accepts_only_explicit_decode_probe_worker(self):
        from .worker_observer import CampaignWorkerExtension
        worker=CampaignWorkerExtension()
        with patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.token_split.install_boundary_teacher') as install:
            with patch.dict(os.environ,{'MLLM_EPD_PREFIX_PROBE':'1','MLLM_EPD_ROLE':'decode',
                    'MLLM_EPD_ALLOW_DECODE_PROBE':'1'}):
                result=worker.campaign_boundary_teacher_install()
                self.assertEqual(result['evidence'],'CORRECTNESS_ONLY')
                install.assert_called_once_with()
            for prefix,role,allow in [('0','decode','1'),('1','decode','0'),('1','prefill','1')]:
                install.reset_mock()
                with patch.dict(os.environ,{'MLLM_EPD_PREFIX_PROBE':prefix,'MLLM_EPD_ROLE':role,
                        'MLLM_EPD_ALLOW_DECODE_PROBE':allow}):
                    with self.assertRaises(AssertionError):worker.campaign_boundary_teacher_install()
                    install.assert_not_called()

    def test_probe_flag_keeps_original_model_compiler_and_connector_command(self):
        from .servers import command
        kwargs=dict(observe=True,ec_transport='cuda_ipc',fixed_reduction=True,deterministic_reductions=True)
        native=command('decode',18842,Path('/dev/shm/probe-contract-unused'),19842,**kwargs)
        probe=command('decode',18842,Path('/dev/shm/probe-contract-unused'),19842,prefix_probe=True,**kwargs)
        self.assertEqual(native,probe)
        connector=json.loads(probe[probe.index('--kv-transfer-config')+1])
        self.assertEqual(connector['kv_connector'],'NixlConnector')
        self.assertEqual(connector['kv_role'],'kv_consumer')
        self.assertNotIn('--enforce-eager',probe)
        self.assertEqual(probe[probe.index('--dtype')+1],'bfloat16')
        with self.assertRaises(AssertionError):
            command('prefill',18841,Path('/dev/shm/probe-contract-unused'),19841,prefix_probe=True,**kwargs)


if __name__=='__main__':unittest.main()
