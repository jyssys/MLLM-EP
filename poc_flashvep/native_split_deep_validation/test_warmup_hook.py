"""Exercise the installed scheduling closure without CUDA initialization."""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


class Marker:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def exists(self):
        return True

    def read_text(self):
        self.reads += 1
        return json.dumps(self.value)


def actual_hook(marker, previous, retire=True):
    tree = ast.parse(Path(__file__).with_name('graph_policy.py').read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'schedule')
    scope = dict(deep_marker=marker, can_schedule=previous, retire_geometry_hook=retire)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_geometry_schedule', 'exec'), scope)
    return scope['schedule']


class WarmupHookTests(unittest.TestCase):
    def owner(self):
        return NS(engine=NS(stage_config=NS(role='decode')), transfers={})

    def test_terminal_state_does_not_leave_repeated_marker_IO(self):
        marker = Marker(dict(state='ALL_COMPLETE'))
        sentinel = object()
        hook = actual_hook(marker, lambda owner, request: sentinel)
        owner = self.owner()
        for _ in range(10):
            self.assertIs(hook(owner, NS()), sentinel)
        self.assertEqual(marker.reads, 1)

    def test_active_barrier_remains_until_every_import_is_ready(self):
        marker = Marker(dict(state='RUNNING', seeds=[7], count=1))
        request = NS(sampling_params=NS(seed=7))
        owner = self.owner()
        transfer = NS(request=request, cancelled=False, ready=False)
        owner.transfers = {'request': transfer}
        hook = actual_hook(marker, lambda owner, request: True)
        self.assertFalse(hook(owner, request))
        transfer.ready = True
        self.assertTrue(hook(owner, request))

    def test_legacy_observer_control_remains_available(self):
        marker = Marker(dict(state='ALL_COMPLETE'))
        owner = self.owner()
        hook = actual_hook(marker, lambda owner, request: True, retire=False)
        for _ in range(10):
            self.assertTrue(hook(owner, NS()))
        self.assertEqual(marker.reads, 10)


if __name__ == '__main__':
    unittest.main()
