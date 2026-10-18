"""Versioned continuation. The v2 evidence remains unchanged."""
from pathlib import Path
from poc_decode_first_depth_aware_dp.contract import *

OLD_ROOT = ROOT
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SPECS = [Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports') / name
         for name in ('decode_first_depth_aware_dp_spec.md', 'decode_first_depth_aware_dp_spec_v3.md')]
BASE_COMMIT = 'e6a350e0a862d94251c5cefa0ef38038b017b8c4'

def read(path):
    """Read local JSON or its lossless published gzip copy."""
    import gzip
    path=Path(path)
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as stream:return json.load(stream)
