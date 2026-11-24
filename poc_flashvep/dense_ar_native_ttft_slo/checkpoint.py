"""Save resumable progress without changing GPU work or scientific gates."""
import json
import time
from .common import ROOT,BASE,BRANCH,START,write,preserve


def main():
    preserve();gates=[];leases=[]
    for path in sorted((ROOT/'state').glob('ns_*_composition_v1.json')):
        value=json.loads(path.read_text());gates.append(dict(path=path.name,label=value.get('label'),passed=value['passed'],error=value.get('error')))
    for path in sorted((ROOT/'state').glob('pair_lease_*.json')):leases.append(json.loads(path.read_text()))
    analyses=[]
    for filename in ('historical_main400_metric_reproduction.json','request_failure_attribution.json',
            'request_critical_paths.json','native_relative_causal_comparisons.json','policy_credit_batch_audit.json',
            'base_source_blob_audit.json','historical_independent_metric_audit.json','cpu_regression_audit.json',
            'arrival_seed_and_trace_audit.json','figure_manifest.json','serial_native_relative_kernel_analysis_v2.json',
            'controlled_intervention_analysis.json','intervention_figure_manifest.json',
            'worker_client_decode_gap_analysis.json','dataset_identity_and_overlap_audit.json',
            'historical_current_runtime_bridge_audit.json','raw_metric_recalculation.json',
            'native_relative_actual_work_audit.json','reused_one_factor_controls.json'):
        path=ROOT/'analysis'/filename
        if path.exists():
            value=json.loads(path.read_text());analyses.append(dict(file=filename,passed=value.get('passed'),
                results=len(value.get('results',[])),starts=len(value.get('starts',[]))))
    controllers={}
    for filename in ('guardian_controller.json','policy_qualification_controllers_v1.json',
            'native_common0_reference_controller_v1.json','serial_native_relative_profile_controller_v2.json',
            'main400_native_relative_controller_v2.json','heldout192_native_relative_controller_v1.json',
            'heldout192_native_relative_controller_v2.json','terminal_analysis_controller_v1.json'):
        path=ROOT/'state'/filename
        if path.exists():controllers[filename]=json.loads(path.read_text())
    value=dict(time=time.time(),status='ACTIVE_ROOT_CAUSE_CAMPAIGN',base=BASE,branch=BRANCH,start_epoch=START,
        deadline_epoch=START+12*3600,last_large_dispatch_epoch=START+10.5*3600,
        initial_qualification=gates,pair_leases=leases,controllers=controllers,completed_analysis=analyses,
        immutable_historical_scope='15 requested Main400 starts reproduced;209 historical timing starts analyzed;191 non-Agg/non-deflection host paths.',
        metric_findings=dict(denominator='400 requests x3 starts =1200 request observations per Main method',
            Layer_TTFT_fail=638,Stock_TTFT_fail=467,Layer_joint_pass=557,Stock_joint_pass=729,
            Layer_ITL_only_fail_requests=5,Layer_ITL_only_fail_output_tokens=3498,
            Stock_ITL_only_fail_requests=4,Stock_ITL_only_fail_output_tokens=462,
            interpretation='TTFT distribution below the5s threshold and token-weighted rare ITL failures both matter. This is metric accounting,not runtime causality.'),
        plan_history='Unlaunched v1 profile/Main CPU controllers were ownership-checked and stopped before GPU dispatch. '
            'Preserve their plans/configs. V2 adds exact NativeCommon0 and forward/reverse/fixed-shuffle timing order. No running GPU experiment stopped.',
        pending_work=[
            'Initial12 qualification and the Common0 addendum all passed. Preserve their13 proofs and all raw.',
            'Serial CPC v2 is complete:Native,Layer,Hybrid each qualifiedN3;Stock s0 diagnostic because ITLp99 observer ratio1.201861;Stock s1/s2 NOT_RUN. No GPU retry.',
            'profile_analysis.py is complete. It uses integer-ns clock mapping after a red/green large-epoch precision fix. Native P/D-first kernels287.920/13.582ms;Layer278.956/23.226;Hybrid237.939/63.326. Serial32 Output16 only,not Main400.',
            'Main400 Natural v2 has13 arms x3 starts and exact historical seeds2026112271/72/73. Required refs:NativeStrong,Common0 for uncapped Stock,Common2 for capped axes.',
            'registry.analysis_runs now includes finished current Main/Heldout timing in separate families. Re-run failure,request_trace,policy_trace,causal and intervention_analysis after timing.',
            'Heldout192 EqualReplay plan has10 arms x3 starts,old seeds2026112281/82/83,11208 tokens. D17,defined before completed Main N3 and any Heldout result,keeps all30 if time fits,else mandatory seven-arm21-job N3 if time fits. The budget ceiling/reserve and effects-blind selection stay unchanged.',
            'Do not add new source-window controls or sweeps. The planned one-factor cap,registration and retirement controls consume the remaining timing budget. Common-D does not match finite P windows.',
            'All five reports now exist as drafts. Historical and serial sections are complete. Final summary and controlled-policy sections still need terminal fresh N3 values,scientific verdict and operational closeout. Do not publish draft status as a completed result.',
            'terminal_analysis.py CPU controller waits for both GPU schedule receipts. It runs the ordered final analysis and verification pipeline. It does not approve reports,choose a verdict,close resources or publish. Inspect its log/result before repeating CPU work.',
            'intervention_analysis.py computes exact one-factor signature contrasts,quality and paired NativeStrong/matching-Common-D decode gates. RegOpen references NativeRegOpen;Stock0 referencesCommon0.',
            'worker/client gap analysis covers80 historical starts. Layer all5 ITL-only failures/3498tokens have worker gaps<=34ms. Stock4/462 likewise. Preserve the client metric;do not claim this proves GC caused them.',
            'Dataset audit passed:Main400 uniquecontexts287,held147/discovery140 disjoint. Calibration32 and Prime8 share neither requests nor contexts with Main or each other.',
            'Runtime bridge audit passed. Fresh arms inherit common buffered I/O,deferred coordinator GC,stable worker paths/fixed ports and P host trace. Do not causally attribute old/new timing differences to one policy.',
            'Installed scheduler predicate audit validates standard491d... and Hybrid dedented eab0... plus the two AST clip sites85/451. Initial false identical-source reader assumption is preserved. Native one-token remainder bypasses bulk-prefill deferral;this is code eligibility,not a measured safe window.',
            'Queue-state audit preserves partial first-window coverage rather than filling it with zero. OldMain D free-block minima:Native1810,Stock392,Layer319,Hybrid1139;no observed zero. P minima can be lower,so do not generalize no-memory-pressure to P.',
            'work_audit.py validates actual request partitions and executed prompt rows. Best Stock8,Layer2 and Hybrid2+1/8 are not iso-work. Legacy target_D_rows3621888 is inherited metadata,not Main400 matching proof. Native adds14400 proxy-recompute rows;all old Main axis actions conserve prompt rows.',
            'reused_controls.py audits15 old same-pair,source,arrival,output-contract one-factor pairs. Credit Heldout N3 is reused rather than duplicated. The GC effect is specific to prior warm Heldout replay,not every old Main tail.',
            'request_trace now separates loop detection,credit acquire and dispatch,with nested identity checks. It records first-query geometry and snapshot queues. causal_analysis now adds descriptive tied-rank feature associations and pass/fail conditional denominators. Re-run after all timing with consistent analysis snapshots.',
            'Recompute raw metrics,CPU tests,figure manifests and PNG/PDF/CSV byte regeneration. Current ten historical figures need final regeneration after updated oracle JSON.',
            'Compact Main request CSVs are implemented but not yet regenerated after the latest field additions. Re-run failure_analysis/request_trace at final. Full CSVs over80MiB receive deterministic verified gzip publication copies.',
            'verification.py now has datasets,arrivals,inventory,metrics,budget and exact figure regeneration. operations.py reuses audited close/archive/private-index publication. completion_audit.py checks terminal scope,not scientific success.',
            'Archive needed new evidence losslessly with SHA256;keep full raw private and preserve prior archives. Publish only new namespace,code and five reports.',
            'Close all owned work,release guardian,restore and verify approved all-eight burners,commit/push private-index branch,verify remote SHA and user checkout/index preservation.'],
        cautions=[
            'Qualified serial GPU costs do not establish exclusive Main400 Natural GPU fractions. Stock capture remains diagnostic;do not promote its point costs.',
            'All observed historical Native paths use official proxy and D first output,not P first-token reuse.',
            'Qualification Natural arrivals use generator2026112401;config seed is the model seed. The separate arrival audit documents this without editing configs.',
            'Current13-arm Main order and10-arm Heldout plan were frozen before timing. Inspect their progress JSON for actual completion.',
            'No dynamic insertion or decode-safe-window success. No speedup is required. Preserve all SLO and correctness failures.',
            'Do not spawn agents. Use apply_patch for source edits. Use the required ASD clarity skill silently for all new reports.'])
    write(ROOT/'state/research_checkpoint.json',value)
    print('RESEARCH_CHECKPOINT',len(gates),'gates',sum(row['passed'] for row in gates),'passed',flush=True)


if __name__=='__main__':main()
