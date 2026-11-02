"""Check the real PD-versus-replica accounting API used by startup requests."""
import types
import unittest
from . import graph_warmup


class Accounting(unittest.TestCase):
    def test_PD_does_not_use_coordinator_counter_schema(self):
        # PDEngine.stats() is coordinator telemetry. Active loop counters live
        # in the native StageWorker stats returned by each peer client.
        client=types.SimpleNamespace(stats=lambda:{'issued_loop_rows':123})
        engine=types.SimpleNamespace(peers={'P':types.SimpleNamespace(client=client)},stats=lambda:{'requests':7})
        self.assertEqual(graph_warmup.worker_stats(engine),{'P':{'issued_loop_rows':123}})

    def test_replica_worker_map_is_preserved(self):
        engine=types.SimpleNamespace(stats=lambda:{'replica0':{'issued_loop_rows':456}})
        self.assertEqual(graph_warmup.worker_stats(engine),{'replica0':{'issued_loop_rows':456}})


if __name__=='__main__':unittest.main()
