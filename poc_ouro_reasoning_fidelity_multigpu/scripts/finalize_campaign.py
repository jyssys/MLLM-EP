"""Validate completion at the declared early-exit boundary."""
import datetime
import subprocess
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    assert read(ROOT/'results/bounded_followup_execution.json')['success']
    assert read(ROOT/'results/final_burner_restore.json')['verification']=='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs'
    assert sha(ROOT/'protocol_lock.json')=='fcb626ccee0a403257117b06974649ae2462d3f5832ab14f8b94c59e0c99c25b'
    lock=read(ROOT/'protocol_lock.json')
    assert sha(ROOT/'results/frozen_questions.json')==lock['inputs_sha256']
    assert sha(ROOT/'sources/authoritative_spec.md')==read(ROOT/'results/environment_public.json')['spec_sha256']
    required=['report.md','reproduce.md','model_tokenizer_eos_audit.md','author_protocol_comparison.md',
        'hf_vllm_fidelity.md','quality_protocol.md','quality_scorecard.md','grader_tests.md',
        'real_workload_contract.md','timing_contract.md','memory_kv_contract.md','prior_overlap.md','advisor_summary.md']
    assert all((ROOT/name).exists() for name in required)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()
    assert branch=='flashvep/ouro-reasoning-fidelity-multigpu-poc'
    quality=read(ROOT/'results/quality_score/native_quality_r4.json')
    decision=read(ROOT/'results/quality_decision.json')
    fidelity=read(ROOT/'results/fidelity_summary.json')
    systems=read(ROOT/'results/exploratory_B1_summary.json')
    assert decision['quality']=='QUALITY_BASELINE_WEAK'
    assert decision['broad_systems_authorized'] is False and decision['method_authorized'] is False
    assert fidelity['reference_runtime_verdict']=='REFERENCE_RUNTIME_PARTIAL'
    assert all(row['completed']==row['correct_completed']==row['total']==6 for row in systems['groups'])
    runs=[]
    for directory in sorted((ROOT/'results').iterdir()):
        if not directory.is_dir():
            continue
        contract=directory/'run_contract.json'
        guard=directory/'guard_status.json'
        if not contract.exists() and not guard.exists() and not (directory/'gpu_policy.json').exists():
            continue
        status=read(guard) if guard.exists() else dict(success=False,status='EARLY_SETUP_REFUSAL_OR_INCOMPLETE_GUARD')
        item=dict(name=directory.name,guard=status,raw_files_preserved=True)
        if contract.exists():
            data=read(contract)
            item.update(mask=data['mask'],command=data['command'],run_contract_sha256=sha(contract))
        runs.append(item)
    ledger=ROOT/'results/run_ledger.json'
    assert not ledger.exists()
    write(ledger,dict(runs=runs,scope='Includes failed setup and initialization attempts. A successful guard does not imply benchmark correctness.'))
    final=ROOT/'results/final_verdict.json'
    assert not final.exists()
    write(final,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        quality=decision['quality'],correctness=decision['correctness'],systems=decision['systems'],
        new_framework_decision='NO_GO_AT_QUALITY_GATE',
        checkpoint='ByteDance/Ouro-2.6B-Thinking',revision=REVISION,R=4,precision='BF16',
        physical_GPUs=GPUS,physical_UUIDs=UUIDS,model_kind='Dense Transformer, not MoE',
        reference_protocol='REFERENCE_PROTOCOL_NOT_IDENTICAL',
        quality_final_lanes=quality['final_lanes'],primary_independent_AIME_problems=30,
        primary_sampled_outputs=90,all_cap_quality_outputs=460,
        original_strict_state_gate=False,functional_confirmation_pass=fidelity['functional_confirmation_pass'],
        fixed_history_diagnostic=read(ROOT/'results/restart_history_gate.json'),
        isolated_sampler_diagnostic=read(ROOT/'results/fixed_logits_sampler_comparison.json'),
        exploratory_B1=systems['groups'],exploratory_independent_problems=2,
        matched_shape_hardware_speedup='NOT_MEASURED',removable_recurrence_specific_cost='UNRESOLVED',
        broad_TP_DP_serving='NOT_RUN: quality gate fails',new_method='NOT_IMPLEMENTED',
        full_quality_under_core_MP0='NOT_RUN',candidate_correctness_and_accuracy='NOT_RUN: no candidate runtime',
        data_availability=read(ROOT/'evidence/manifest.json'),
        validation='15 implementation tests pass. The separate historical scientific property remains a failed test.',
        final_burners='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs',
        branch=branch,branch_parent=read(ROOT/'results/environment_public.json')['git_parent'],
        containing_commit='Use git log -1 --format=%H -- poc_ouro_reasoning_fidelity_multigpu',
        report='poc_ouro_reasoning_fidelity_multigpu/report.md',
        next_action='Establish a separately frozen complete-output quality baseline with verified scheduling and reference fidelity before broad TPxDP profiling.',
        limitations=['The public evaluator differs from the private author protocol.',
            'AIME censoring persists at the largest frozen cap.',
            'Functional agreement does not establish strict state equivalence.',
            'The bootstrap does not quantify between-session runtime variation.',
            'The small hardware audit is not a full serving or critical-path campaign.',
            'Off-Git state arrays and the unredacted environment archive remain local, not remotely published.']))
    print('Validated quality-gated completion, independent verdict axes, and restored assigned burners.')


if __name__=='__main__':
    main()
