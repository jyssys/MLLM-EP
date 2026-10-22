# Capacity status

Qualified capacity is NOT_RUN.
No new 90%-attainment rate is reported.
The deadline controller did not pass its prerequisite gate.

The specification requires at least 1,000 offered requests per arm/rate and three independent server starts.
It also requires stable queues and confidence-aware attainment.
The 32-token cost cohorts do not satisfy these requirements.
Their drained tokens/s values are not sustainable capacity.
Warm-up, bounded generation, event observation, and finite arrival windows remain separate labels.

The future capacity protocol must preserve offered requests in its denominator.
It must report TTFT, every-gap TBT, joint attainment, backlog slope, and drain time separately.
It must use common Poisson/burst arrivals across policies.
The final online families remain untouched for that test.
