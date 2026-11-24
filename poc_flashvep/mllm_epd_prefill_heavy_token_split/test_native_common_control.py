"""Test the real Native launch flags and the recomputed last-query gate."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .common import write
from . import ipc_ec_control as ipc
from .native_common_control import compare_native_state


class NativeCommonControlTest(unittest.TestCase):
    def test_actual_native_launcher_changes_both_llm_workers_not_encoder_or_adapter(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir()
            calls=[];tag='cpu_common_flags_fixture'
            def run(args,**kwargs):
                calls.append(args)
                name=args[args.index('--name')+1]
                write(root/'state'/f'server_{name}.json',dict(executed_source_cwd=directory))
            process=SimpleNamespace(pid=12345,poll=lambda:None)
            class Response:
                status=200
                def __enter__(self):return self
                def __exit__(self,*args):pass
            owned=[]
            with patch.object(ipc,'ROOT',root),patch.object(ipc,'wait_ready'),\
                 patch.object(ipc.subprocess,'run',side_effect=run),\
                 patch.object(ipc.subprocess,'Popen',return_value=process),\
                 patch.object(ipc.urllib.request,'urlopen',return_value=Response()):
                names,_=ipc.launch(tag,'cuda_ipc',True,True,owned_names=owned)
            self.assertIs(names,owned)
            self.assertEqual(len(owned),3)
            for command in calls:
                role=command[command.index('--role')+1]
                self.assertNotIn('--token-split',command)
                self.assertEqual('--fixed-reduction' in command,role!='encoder')
                self.assertEqual('--deterministic-reductions' in command,role!='encoder')

    def fixture(self):
        length=4
        prompt=torch.arange(length);positions=torch.arange(length).repeat(3,1)
        values={}
        records={}
        for role in ['T','P','D']:
            values[role]=dict(prompt_token_ids=prompt.clone(),full_prompt_mrope=positions.clone(),
                positions_0=positions[...,-1:].clone() if role=='D' else positions.clone(),
                inputs_embeds_0=torch.ones(1 if role=='D' else length,3))
            values[role].update({f'prompt_KV_{i}':torch.ones(length,1,2) for i in range(28)})
            records[role]=dict(snapshot=dict(mrope_delta=0,event=dict(index=0,
                start=length-1 if role=='D' else 0,query_tokens=1 if role=='D' else length)))
        return values,records

    def test_last_query_is_checked_separately_and_import_is_byte_exact(self):
        values,records=self.fixture()
        self.assertTrue(compare_native_state(values,records)['passed'])
        changed=copy.deepcopy(values)
        changed['D']['prompt_KV_27'][-1,0,0]+=1
        result=compare_native_state(changed,records)
        self.assertFalse(result['passed'])
        self.assertTrue(result['layers'][-1]['imported_prefix_strict'])
        self.assertFalse(result['layers'][-1]['last_query_tolerance_pass'])
        changed=copy.deepcopy(values)
        changed['D']['prompt_KV_0'][0,0,0]+=.001
        self.assertFalse(compare_native_state(changed,records)['passed'])


if __name__=='__main__':unittest.main()
