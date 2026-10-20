"""Freeze inputs and preserve the experiment contract before GPU measurements."""
import datetime
import subprocess
from .contract import *
from .model_adapter import tokenizer


def main():
    t = tokenizer()
    old = json.loads((PROJECT / 'poc_depth_cdb_factorial/artifacts/workload_frozen.json').read_text())
    groups = {}
    for length in [16, 128, 512, 2048]:
        rows = []
        for i, r in enumerate(old['speed'][:32]):
            ids = t.encode(r['prompt'], add_special_tokens=False)
            # Controlled token-shape workload, not a benchmark quality claim.
            ids = (ids * (1 + length // len(ids)))[:length]
            rows.append(dict(id=f'request-{i}', source=r['id'], input_ids=ids))
        groups[str(length)] = rows
    write(ROOT / 'workload.json', dict(seed=42, groups=groups,
        scope='Real prompt token streams repeated/truncated to fixed hardware shapes. No quality claim.',
        prompt_chunking=False, output_stopping='Forced length primary; EOS smoke separately.',
        frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write(ROOT / 'preflight.json', dict(spec_path=str(SPEC), spec_sha256=sha(SPEC),
        base_git_SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        model=str(MODEL), model_revision=REVISION, model_license='Apache-2.0, research only model card',
        source_sha256={name:sha(MODEL/name) for name in ['config.json','modeling_ouro.py','configuration_ouro.py','README.md']},
        numeric_limits=TOLERANCE, physical_GPUs=GPUS,
        runtime='Pinned official HF decoder/SDPA. Four CUDA worker threads and device-to-device peer copies.',
        process_layout='One process with four independent FIFO workers, one CUDA stream/device per worker. Not torchrun/NCCL.',
        head_plan='Same embedding/norm/gate/LM head replicas on every GPU in both placements.',
        correctness_first=True, dynamic_depth='Not implemented before fixed-depth gates pass.',
        pushes='User final instruction requests GitHub push after completion.'))
    (ROOT / 'spec_frozen.md').write_text(SPEC.read_text())
    print('FROZEN', ROOT)


if __name__ == '__main__':
    main()
