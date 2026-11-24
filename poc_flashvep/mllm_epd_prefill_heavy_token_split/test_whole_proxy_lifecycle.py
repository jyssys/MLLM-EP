"""The real controller must not reuse the native proxy log for its whole proxy."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import whole_deflection_control as control


class WholeProxyLifecycleTest(unittest.TestCase):
    def test_correctness_controller_preserves_original_native_proxy_log(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['state','logs','data','analysis']:(root/name).mkdir()
            (root/'data/calibration_manifest_template_v2.json').write_text(json.dumps(dict(requests=[{}]*32)))
            native=SimpleNamespace(pid=100,poll=lambda:None)
            whole=SimpleNamespace(pid=200,poll=lambda:None)
            tag=control.TAG+'_CORRECT';name=tag+'_E4'
            path=root/'logs'/f'{tag}_proxy.log';original=b'Original closed Native proxy log. Do not replace.\n'
            def launch(*args,**kwargs):
                path.write_bytes(original)
                (root/'state'/f'server_{name}.json').write_text(json.dumps(dict(executed_source_cwd=directory)))
                return [name],native
            class Health(io.BytesIO):status=200
            async def cohort(*args):return []
            with patch.object(control,'ROOT',root),patch.object(control,'launch',side_effect=launch),\
                    patch.object(control,'stop_native_proxy'),patch.object(control,'stop_whole'),\
                    patch.object(control,'retire'),patch.object(control,'set_common_ipc_policy'),\
                    patch.object(control,'rpc',return_value=[dict(inline_role=None)]),\
                    patch.object(control,'correctness_cohort',side_effect=cohort),\
                    patch.object(control.subprocess,'Popen',return_value=whole),\
                    patch.object(control.urllib.request,'urlopen',side_effect=lambda *a,**k:
                        Health(json.dumps(dict(name=tag+'_whole_proxy')).encode())):
                control.correctness()
            self.assertEqual(path.read_bytes(),original)
            self.assertTrue((root/'logs'/f'{tag}_whole_proxy.log').is_file())


if __name__=='__main__':unittest.main()
