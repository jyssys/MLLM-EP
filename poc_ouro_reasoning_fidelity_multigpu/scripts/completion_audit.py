"""Read-only completion checks and a compact owned evidence record."""
import datetime
import subprocess
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def main():
    assert read(ROOT/'results/published_replay_verification.json')['success']
    assert read(ROOT/'results/final_burner_restore.json')['verification']=='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs'
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=PROJECT,text=True).splitlines()
    assert staged and all(path.startswith(ROOT.name+'/') for path in staged)
    whitespace=subprocess.run(['git','diff','--cached','--check','--',ROOT.name,
        ':(exclude)'+ROOT.name+'/results/**',':(exclude)'+ROOT.name+'/sources/**',
        ':(exclude)'+ROOT.name+'/evidence/**'],cwd=PROJECT,text=True,capture_output=True)
    assert whitespace.returncode==0,whitespace.stdout
    tests=subprocess.run([CPU,'-m','unittest',*['poc_ouro_reasoning_fidelity_multigpu.tests.'+name
        for name in ['test_contract','test_logit_metrics','test_problem_clusters','test_guard_scope']],'-v'],
        cwd=PROJECT,text=True,capture_output=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    selected=[]
    for line in processes.splitlines():
        uuid,pid=[value.strip() for value in line.split(',')]
        if uuid.removeprefix('GPU-') not in UUIDS:
            continue
        worker,argv=burner_worker_ancestor(int(pid))
        gpu=GPUS[UUIDS.index(uuid.removeprefix('GPU-'))]
        assert f'--gpu {gpu} ' in argv
        selected.append(dict(physical_GPU=gpu,context_PID=int(pid),burner_worker_PID=worker))
    assert sorted(row['physical_GPU'] for row in selected)==GPUS
    target=ROOT/'results/completion_audit.json'
    assert not target.exists()
    critical=['protocol_lock.json','results/frozen_questions.json','results/final_verdict.json',
        'results/quality_score/native_quality_r4.json','results/quality_score/quality_generations.csv',
        'evidence/manifest.json','evidence/natural_generations_v1.tar.gz','report.md','advisor_summary.md']
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip(),
        staged_owned_files=len(staged),staged_paths_scoped=True,authored_whitespace_check_exit=0,
        raw_whitespace_policy='Preserve raw logs, CSV line endings, and the copied authoritative specification byte for byte.',
        implementation_test_exit=tests.returncode,implementation_test_output=tests.stdout+tests.stderr,
        historical_scientific_property_exit=read(ROOT/'results/validation_execution_v3.json')['scientific_assumption']['exit_code'],
        all_tests_pass=False,quality_archive_replay_pass=True,selected_burner_contexts=selected,
        experiment_GPU_contexts_remaining=False,assigned_burners_active=True,
        GPU_state=subprocess.check_output(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
        critical_file_sha256={name:sha(ROOT/name) for name in critical},
        original_worktree_policy='Only this new worktree prefix is staged. No reset, clean, or unrelated-file staging.'))
    print('Completion checks pass. Only owned files are staged. Assigned GPUs contain verified project burners.')


if __name__=='__main__':
    main()
