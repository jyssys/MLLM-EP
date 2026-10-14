from types import SimpleNamespace
import unittest
from poc_dllm_prefix_canvas_cp.graph_events import GraphEvents


class GraphEventTests(unittest.TestCase):
    def test_shared_RoPE_module_is_patched_once(self):
        obj=SimpleNamespace(forward=lambda:'native')
        bank=GraphEvents(detail=True)
        bank.patch(obj,'forward',lambda:'first')
        bank.patch(obj,'forward',lambda:'second')
        self.assertEqual(obj.forward(),'first')
        self.assertEqual(len(bank.patches),1)
        bank.restore();self.assertEqual(obj.forward(),'native')


if __name__=='__main__':unittest.main()
