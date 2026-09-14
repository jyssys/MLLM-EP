# Online serving harness

The headline harness is route-replay EP-stage serving, not isolated calls. Each
request follows the chronological GSM8K or HumanEval M trajectory. Service-time
curves come from same-node GPU measurements, and the legal two-slot factor comes
from the three-restart real-route concurrency experiment.

Workloads are Poisson arrivals at calibrated 30%, 60%, 82.5%, and 95% offered
load; bursty ON/OFF arrivals; and closed-loop concurrency 1/2/4/8/16/32. Five
random seeds are used. Reported request latency includes EP-stage service and
EP queueing only. It does **not** include attention, host tokenization, or a live
continuous-batching scheduler, so it is not mislabeled as full-model E2E.

The real LLaDA2 online engine validation was gated on a promoted structural
result. Because route-replay O4 remained below 8%, loading a second full-model
online harness and a custom CUDA path would not change the decision and was
skipped by contract.

The simulator and all calibration constants are in
`scripts/analyze_and_simulate_serving.py`; raw per-seed rows are in
`SERVING_RESULTS.csv`.
