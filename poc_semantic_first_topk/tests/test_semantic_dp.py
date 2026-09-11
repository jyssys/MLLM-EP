from poc_semantic_first_topk.compose_semantic_oracle import allocation


def rows():
    result = []
    # The non-monotonic optimum requires jumping directly from K=8 to K=3.
    for group, size in ((0, 2), (1, 2)):
        for k in range(1, 8):
            risk = 0.1 * (8 - k)
            if group == 0 and k == 3:
                risk = 0.01
            result.append({"group": group, "group_size": size, "k": k,
                           "kl_per_answer_token": risk})
    return result


def test_multiple_choice_dp_handles_nonmonotonic_states():
    result = allocation(rows(), .3125, set(range(1, 9)))
    assert result["dropped"] == 10
    assert result["group_k"][0] == 3
    assert result["additive_kl_oracle"] == .01


def test_full_grid_weakly_dominates_coarse_at_shared_budget():
    coarse = allocation(rows(), .25, {1, 2, 4, 6, 8})
    full = allocation(rows(), .25, set(range(1, 9)), exact_drop=coarse["dropped"])
    assert full["dropped"] == coarse["dropped"]
    assert full["additive_kl_oracle"] <= coarse["additive_kl_oracle"]
