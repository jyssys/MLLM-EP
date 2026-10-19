# Run from the root of the new worktree. Do not change the allowed GPU set.
set -euo pipefail
# Environment and package setup reuse the pinned v3 reproduction instructions.
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.freeze
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.prepare_plans
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.trace_plans
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.environment
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.gpu_session --gpus 4,5,6,7 --output outputs/smoke_validation_v1 --timeout 1800 -- /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.run --plan poc_depth_cdb_factorial/artifacts/smoke_plan.json --validate --trace
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.campaign --phase speed
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.extend_repeats
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.freeze_order_control
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.campaign --phase all
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.analyze --quality
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.trace_analysis
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.audit_v3
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.analyze_sensitivities
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.equal_progress
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.figures
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.publish_evidence
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest -v poc_depth_cdb_factorial.tests.test_contract
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_depth_cdb_factorial.scripts.final_audit
