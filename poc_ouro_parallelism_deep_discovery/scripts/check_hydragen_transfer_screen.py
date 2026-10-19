"""CPU-only algebra checks. These checks do not model kernel time or traffic."""
import json
import math
from collections import Counter, defaultdict


def partial(scores, values):
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return sum(w * v for w, v in zip(weights, values)) / denominator, maximum + math.log(denominator)


def merge(parts):
    maximum = max(lse for _, lse in parts)
    weights = [math.exp(lse - maximum) for _, lse in parts]
    return sum(out * weight for (out, _), weight in zip(parts, weights)) / sum(weights)


def grouping(rows):
    counts = Counter(group for group, _ in rows)
    owners = defaultdict(set)
    for group, owner in rows:
        owners[group].add(owner)
    return dict(queries=len(rows), groups=len(counts), group_widths=sorted(counts.values()),
                independent_logical_prefix_units=len(rows),
                grouped_logical_prefix_units=len(counts),
                replica_group_owner_units=sum(len(value) for value in owners.values()),
                tp_grouped_cluster_units=len(counts),
                logical_duplicate_units=len(rows) - len(counts))


def main():
    errors = []
    prefix_k, prefix_v = [0.5, -1.0], [2.0, -3.0]
    for query, unique_k, unique_v in [(0.2, [0.25], [4.0]), (1.3, [-0.75, 0.2], [-2.0, 1.0])]:
        prefix_scores = [query * key for key in prefix_k]
        suffix_scores = [query * key for key in unique_k]
        whole, _ = partial(prefix_scores + suffix_scores, prefix_v + unique_v)
        decomposed = merge([partial(prefix_scores, prefix_v), partial(suffix_scores, unique_v)])
        error = abs(whole - decomposed)
        assert error < 1e-14
        errors.append(error)
    cases = {
        'one_query_at_each_depth': [(depth, depth) for depth in range(4)],
        'four_queries_same_depth_split_replicas': [(0, owner) for owner in range(4)],
        'two_branches_at_four_depths_branch_owned': [(depth, branch) for depth in range(4) for branch in range(2)],
        'two_branches_at_four_depths_depth_owned': [(depth, depth) for depth in range(4) for _ in range(2)],
    }
    results = {name: grouping(rows) for name, rows in cases.items()}
    assert results['one_query_at_each_depth']['logical_duplicate_units'] == 0
    assert results['four_queries_same_depth_split_replicas']['replica_group_owner_units'] == 4
    assert results['four_queries_same_depth_split_replicas']['tp_grouped_cluster_units'] == 1
    assert results['two_branches_at_four_depths_branch_owned']['replica_group_owner_units'] == 8
    assert results['two_branches_at_four_depths_depth_owned']['replica_group_owner_units'] == 4
    print(json.dumps(dict(status='PASS', scalar_decomposition_cases=2,
        scalar_absolute_errors=errors, grouping_cases=results,
        arithmetic='Python binary64 scalar algebra, not BF16 execution',
        unit='One equally sized, immutable prefix block per depth',
        interpretation='Logical source-block accounting only. No HBM, runtime, or ownership-optimum measurement.',
        gpu_execution=False, third_party_imports=False), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
