from collections import Counter
import random
from poc_dllm_block_local_replica.scripts.destination_closure_oracle import bundle_greedy as reference
from poc_dllm_request_persistent_replica.fast_bundle import bundle_greedy


def test_exact_reference_parity():
    rng = random.Random(33)
    for case in range(25):
        groups = {l: Counter() for l in range(1, 5)}
        for l in groups:
            for _ in range(40):
                experts = rng.sample(range(16), rng.randint(1, 4))
                groups[l][(experts[0] // 4, sum(1 << e for e in experts))] += rng.randint(1, 9)
        slots = rng.randint(1, 35)
        assert bundle_greedy(groups, slots) == reference(groups, slots)
        # Deterministic load-guard rejection, including repeated bundle offers.
        allowed = lambda layer, experts: (sum(experts) + layer) % 5 != 0
        assert bundle_greedy(groups, slots, allowed) == reference(groups, slots, allowed)
