from poc_dllm_attn_moe_pipeline.scripts.build_dependency_dag import critical_path, boundary_dag
import pytest


def test_current_waits_all_groups_and_no_double_count():
    nodes = boundary_dag([100,100],[30,30],20,5,10,True)
    total,finish = critical_path(nodes)
    assert finish['current'] >= finish['q0'] and finish['current'] >= finish['q1']
    assert total == 265
    serial,_ = critical_path(boundary_dag([100,100],[30,30],20,5,10,False))
    assert serial == total  # Unlimited-resource lower bound still pays tail QKV.


def test_unknown_is_not_free():
    with pytest.raises(ValueError):
        critical_path([dict(node_id='layout',predecessors=[],duration_us=None)])


def test_missing_dependency_rejected():
    with pytest.raises(ValueError):
        critical_path([dict(node_id='next',predecessors=['future'],duration_us=1)])
