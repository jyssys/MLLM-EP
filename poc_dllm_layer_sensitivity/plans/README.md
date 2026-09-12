# Policy plans

Plans are generated with `scripts/generate_policy_plans.py`. Every causal plan
starts with a baseline generation under the same loaded model, followed by
single-layer interventions separated by phase.
