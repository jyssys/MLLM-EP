"""The Native child must resolve its own compiler and remain batch invariant."""
import shutil,unittest
from .native import environment


class EnvironmentTests(unittest.TestCase):
    def test_child_compiler_is_available(self):
        self.assertIsNotNone(shutil.which('ninja',path=environment([0])['PATH']))

    def test_native_flags_and_gpu_map(self):
        env=environment([4])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4')
        self.assertEqual(env['VLLM_BATCH_INVARIANT'],'1')


if __name__=='__main__':unittest.main()
