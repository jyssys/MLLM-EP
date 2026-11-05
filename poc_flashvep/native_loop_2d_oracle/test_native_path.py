"""CPU regressions for orchestration seams. Not GPU numerical qualification."""
import ast
import unittest
import pickle
from pathlib import Path


class ClosureTests(unittest.TestCase):
    def test_distinct_constructor_closures(self):
        source = Path(__file__).with_name('native_path.py').read_text()
        tree = ast.parse(source)
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name, required in (('init', 'initialize_pd'), ('stage', 'initialize_stage')):
            calls = [node.func.id for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            self.assertIn(required, calls)
            self.assertNotIn('initialize', calls)

    def test_factory_survives_standard_pickle(self):
        from .native_path import NativeFactory
        method = pickle.loads(pickle.dumps(NativeFactory.create))
        self.assertEqual(method, NativeFactory.create)

    def test_connector_selected_at_actual_constructor(self):
        tree = ast.parse(Path(__file__).with_name('native_path.py').read_text())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'init')
        self.assertIsInstance(init.body[0], ast.Assign)
        self.assertEqual(init.body[0].value.id, 'NativeFactory')


if __name__ == '__main__':
    unittest.main()
