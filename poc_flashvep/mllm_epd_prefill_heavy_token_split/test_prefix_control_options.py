"""Run the actual control flow without starting any GPU process."""
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class PrefixControlOptionsTest(unittest.TestCase):
    def run_control(self,extra_options,expected_arms):
        from . import prefix_quantum_correctness_control as control
        from .common import write
        with TemporaryDirectory() as folder:
            root=Path(folder)
            row=dict(request_id='original',total_input_tokens=2961,image_count=2,prompt_SHA256='a'*64)
            write(root/'data/calibration_manifest_template_v2.json',dict(requests=[row]*7))
            for role,gpu in [('E',4),('P',5),('D',6)]:
                write(root/'state'/f'server_token_NUMERICAL_C24_{role}{gpu}.json',
                    dict(finished=True,guardian=999999999))
            with patch.multiple(control,ROOT=root,TAG=control.TAG,EC=control.EC,E=control.E,HE=control.HE,HD=control.HD),\
                 patch.object(control,'preserve'),patch.object(control,'snapshot',return_value={'archive_SHA256':'b'*64}),\
                 patch.object(control,'experiment_deadline',return_value=time.time()+3600),\
                 patch.object(control.subprocess,'run'),patch.object(control,'wait_ready'),\
                 patch.object(control,'rpc'),patch.object(control,'info',return_value={}),\
                 patch.object(control,'launch_proxy',side_effect=lambda *a,**k:(object(),'http://unused')),\
                 patch.object(control,'stop_proxy'),patch.object(control,'captured'),\
                 patch.object(control,'compare',return_value={'strict_pass':True}),\
                 patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.prefix_quantum_cost.measure') as measured,\
                 patch('sys.argv',['test','--tag','prefix_QUANTA_C26_TEST','--measure-quanta']+extra_options):
                control.main()
            self.assertEqual(measured.call_count,len(expected_arms))
            self.assertEqual([call.args[1] for call in measured.call_args_list],expected_arms)

    def test_server_command_lists_do_not_replace_cost_options(self):
        self.run_control([],['E','D'])

    def test_d_only_followup_does_not_repeat_closed_e_costs(self):
        self.run_control(['--cost-arm','D'],['D'])


if __name__=='__main__':unittest.main()
