"""Reproduce the actual official evaluator backend guard without a model load."""
import sys
from .common import XLLM
sys.path.insert(0,str(XLLM))
from xllm.paper_part1.evaluator import _selected_flash_binary_identity

if __name__=='__main__':
    print(_selected_flash_binary_identity())
