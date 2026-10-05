"""CPU reference tests; GPU native-state checks are in component evidence."""
import subprocess
import sys
from poc_dllm_attn_moe_pipeline.contract import PROJECT

if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',
        'poc_dllm_attn_moe_pipeline/tests/test_attention_state_merge.py'],cwd=PROJECT))
