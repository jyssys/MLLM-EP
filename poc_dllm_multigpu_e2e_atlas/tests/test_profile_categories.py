import unittest
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import classify,replay_count


class ProfileCategoryTests(unittest.TestCase):
    def test_opaque_graphs_are_real_replay_occurrences(self):
        opaque=[dict(node=None,name='NATIVE_CUDA_GRAPH_OPAQUE',correlation=n) for n in range(15)]
        self.assertEqual(replay_count(opaque),15)
        nodes=[dict(node=n,name='compute',correlation=7) for n in range(40)]
        self.assertEqual(replay_count(nodes),1)

    def test_actual_allreduce_is_not_ep_dispatch(self):
        self.assertEqual(classify('ncclDevKernel_AllReduce_Sum_bf16_RING_LL'),'TP_ALL_REDUCE')
        self.assertEqual(classify('memcpy128'),'IN_DEVICE_COPY_KERNEL_UNRESOLVED')


if __name__=='__main__':unittest.main()
