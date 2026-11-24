"""The numerical control keeps compiled defaults untouched unless selected."""
import json
from pathlib import Path
import unittest

from .servers import command


class FixedReductionTest(unittest.TestCase):
    def test_original_default_and_explicit_compiled_control(self):
        for role in ['prefill','decode','pd','agg']:
            common=dict(observe=True)
            original=command(role,18000,Path('/dev/shm/cpu_command_only'),19000,**common)
            changed=command(role,18000,Path('/dev/shm/cpu_command_only'),19000,fixed_reduction=True,**common)
            self.assertNotIn('--enforce-eager',changed)
            self.assertNotIn('--compilation-config',original)
            config=json.loads(changed[changed.index('--compilation-config')+1])
            self.assertEqual(config,dict(inductor_compile_config={'dynamic_scale_rblock':False}))

    def test_installed_torch_and_vllm_accept_the_actual_key(self):
        import torch._inductor.config as config
        from vllm.config import CompilationConfig
        settings={'dynamic_scale_rblock':False}
        value=CompilationConfig(inductor_compile_config=settings)
        self.assertFalse(value.inductor_compile_config['dynamic_scale_rblock'])
        with config.patch(settings):self.assertFalse(config.dynamic_scale_rblock)

    def test_vision_and_eager_modes_cannot_silently_receive_the_control(self):
        for role in ['encoder','reference']:
            with self.assertRaises(AssertionError):
                command(role,18000,Path('/dev/shm/cpu_command_only'),19000,observe=True,
                    eager_reference=role=='reference',fixed_reduction=True)


if __name__=='__main__':unittest.main()
