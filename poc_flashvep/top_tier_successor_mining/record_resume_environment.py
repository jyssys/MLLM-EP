"""Record source/config and hardware mapping without launching GPU kernels."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    devices=allowed_devices()
    task=Path(__file__).resolve().parent
    paths=[task/"GPU_EXECUTION_POLICY.json",task/"gpu_scope.py",
           task/"run_ep_serving.py",task/"ep_hooks/successor_hook.py",
           task/"modes_frontier.py",task/"libra_native_sanity.py",
           Path("/home/esjung/vllm-ep/run_utilize.sh")]
    paths += [Path("/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm")/p
              for p in ("inputs/llm.py","v1/core/encoder_cache_manager.py",
                        "model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py")]
    snapshot={"utc":datetime.now(timezone.utc).isoformat(),"physical_gpus":devices,
              "hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
              "missing_optional_source_paths":[str(p) for p in paths if not p.exists()],
              "gpu_mapping":subprocess.check_output(["nvidia-smi","-i",",".join(map(str,devices)),
                    "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version","--format=csv"],text=True),
              "topology":subprocess.check_output(["nvidia-smi","topo","-m"],text=True),
              "burn_is_research":False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(snapshot,indent=2))
    print(json.dumps({"out":str(args.out),"devices":devices,"source_files":len(snapshot["hashes"])}))


if __name__=="__main__":main()
