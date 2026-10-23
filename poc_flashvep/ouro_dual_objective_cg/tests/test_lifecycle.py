"""Catch retained-engine references at the actual measurement wrapper seam."""
import gc
from types import SimpleNamespace
import unittest
import weakref
from ..work_ledger import WorkLedger

class EngineFixture:
    def __init__(self):
        self.executor=SimpleNamespace(submit=lambda batch:batch)

class LifecycleTests(unittest.TestCase):
    def test_closed_ledger_does_not_retain_engine(self):
        engine=EngineFixture();weak=weakref.ref(engine)
        ledger=WorkLedger(engine);ledger.close()
        del engine;gc.collect()
        self.assertIsNone(weak(),'Closed accounting wrapper retains the KV-owning engine')

if __name__=='__main__':unittest.main()
