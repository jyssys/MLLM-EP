"""Use the installed CPU reduction heuristic, not a simulated GPU result."""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from .servers import command


class DeterministicReductionTest(unittest.TestCase):
    def test_actual_combo_path_does_not_request_a_banned_benchmark(self):
        import torch._inductor.config as config
        from torch._inductor.runtime.benchmarking import may_ban_benchmarking
        from torch._inductor.scheduler import Scheduler
        from vllm.config import CompilationConfig
        args=command('prefill',18000,Path('/dev/shm/cpu_command_only'),19000,
            observe=True,fixed_reduction=True,deterministic_reductions=True)
        options=CompilationConfig(**json.loads(args[args.index('--compilation-config')+1]))
        settings=options.inductor_compile_config
        self.assertTrue(settings['combo_kernels'],'Keep horizontal fusion unchanged')
        called=[]
        def benchmark(nodes):called.append('unvetted');may_ban_benchmarking()
        scheduler=SimpleNamespace(_any_atomic_add=lambda nodes:False,benchmark_fused_nodes=benchmark)
        node=SimpleNamespace(get_device=lambda:'cuda:0',get_nodes=lambda:[])
        with config.patch(settings):self.assertTrue(Scheduler.speedup_by_combo_kernel(scheduler,[node]))
        self.assertEqual(called,[])

    def configs(self,deterministic=False):
        from torch._inductor.runtime.triton_heuristics import reduction
        from torch._inductor.runtime.hints import ReductionHint
        return reduction(dict(x=8192,r0_=4096),reduction_hint=ReductionHint.INNER,
            triton_meta=dict(signature={}),return_configs=True,
            inductor_meta=dict(num_load=5,num_reduction=1,add_persistent_rblock=True,
                dynamic_scale_rblock=False,deterministic=deterministic))

    def test_dynamic_scale_disabled_still_has_two_initial_reduction_tiles(self):
        self.assertEqual({c.kwargs['R0_BLOCK'] for c in self.configs()},{2048,4096})

    def test_deterministic_mode_has_one_installed_heuristic_choice(self):
        values=self.configs(True)
        self.assertEqual(len(values),1)
        self.assertEqual(values[0].kwargs,dict(XBLOCK=1,R0_BLOCK=2048))

    def test_common_compiled_policy_does_not_change_original_default(self):
        from vllm.config import CompilationConfig
        import torch._inductor.config as config
        for role in ['prefill','decode','agg','pd']:
            changed=command(role,18000,Path('/dev/shm/cpu_command_only'),19000,observe=True,
                fixed_reduction=True,deterministic_reductions=True)
            self.assertNotIn('--enforce-eager',changed)
            settings=json.loads(changed[changed.index('--compilation-config')+1])['inductor_compile_config']
            self.assertEqual(settings,dict(dynamic_scale_rblock=False,deterministic=True,
                combo_kernels=True,benchmark_combo_kernel=False))
            with config.patch(settings):self.assertTrue(config.deterministic)
            self.assertNotEqual(CompilationConfig().compute_hash(),CompilationConfig(inductor_compile_config=settings).compute_hash())
        original=command('prefill',18000,Path('/dev/shm/cpu_command_only'),19000,observe=True)
        self.assertNotIn('--compilation-config',original)
        with self.assertRaises(AssertionError):
            command('encoder',18000,Path('/dev/shm/cpu_command_only'),19000,observe=True,deterministic_reductions=True)


if __name__=='__main__':unittest.main()
