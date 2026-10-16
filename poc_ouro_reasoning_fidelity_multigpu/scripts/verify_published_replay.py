"""Rescore the public archive in a new temporary directory. No GPU work."""
import os
import subprocess
import tarfile
import tempfile
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    manifest=read(ROOT/'evidence/manifest.json')['natural_outputs']
    archive_path=ROOT/'evidence/natural_generations_v1.tar.gz'
    assert sha(archive_path)==manifest['sha256']
    expected={row['relative_path']:row['sha256'] for row in manifest['files']}
    with tempfile.TemporaryDirectory(prefix='ouro-score-replay-') as temporary:
        directory=Path(temporary)
        with tarfile.open(archive_path,'r:gz') as archive:
            assert {row.name for row in archive.getmembers()}==set(expected)
            for member in archive.getmembers():
                assert member.isfile() and member.name.startswith('results/')
                assert '..' not in Path(member.name).parts
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected[member.name]
            archive.extractall(directory,filter='data')
        result=subprocess.run([CPU,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning',
            '--source',str(directory/'results/quality_v1'),'--output',str(directory/'score')],
            cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        reproduced=read(directory/'score/native_quality_r4.json')
        original=read(ROOT/'results/quality_score/native_quality_r4.json')
        for name in ['all_caps','final_lanes','quality_gate','scorer','reference_protocol']:
            assert reproduced[name]==original[name],name
        csv_sha=sha(directory/'score/quality_generations.csv')
        assert csv_sha==sha(ROOT/'results/quality_score/quality_generations.csv')
        output=ROOT/'results/published_replay_verification.json'
        assert not output.exists()
        write(output,dict(success=True,GPU_used=False,archive_sha256=sha(archive_path),
            raw_wave_files_verified=len(expected),quality_generations=460,
            score_fields_and_problem_bootstrap_identical=True,quality_csv_identical=True,
            quality_csv_sha256=csv_sha,scorer_stdout=result.stdout,
            scope='Reproduces grading from preserved public outputs. It does not reproduce model generation or between-session numerical behavior.'))
    print('Public-archive CPU replay passes. Scores, bootstrap fields, and the full grading CSV match.')


if __name__=='__main__':
    main()
