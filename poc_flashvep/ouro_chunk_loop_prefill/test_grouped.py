"""CPU state and numerical tests. Random tiny weights are unit tests only."""

from dataclasses import replace
import pytest
import torch
from tests.helpers import tiny_ouro_model
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import RequestPhase
from vllm_rlt.worker.executor import build_execution_batch


def execute(model, group, length, chunk, budget):
    engine = StageEngine(
        model,
        cache_config=CacheConfig(256, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=budget,
            prefill_chunk_size=chunk,
            prefill_loop_group_size=group,
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
    )
    for rid, ids in [("a", list(range(1, length + 1))), ("b", [8, 7, 6])]:
        engine.add_request(rid, ids, SamplingParams(max_tokens=3, max_loops=4))
    work = {rid: [0] * 4 for rid in ("a", "b")}
    seen = []
    outputs = {}
    for _ in range(2000):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            if out.finished:
                outputs[out.request_id] = out.token_ids
        batch = engine.last_schedule
        if batch.phase == RequestPhase.PREFILLING_RECURRENT:
            bound = build_execution_batch(batch, 4)
            seen.append(bound.loop_count)
            for req in batch.requests:
                for d in range(req.depth, req.depth + req.loop_count):
                    work[req.request.request_id][d] += req.token_count
    else:
        pytest.fail("No scheduler progress")
    assert work == {"a": [length] * 4, "b": [3] * 4}
    assert seen and max(seen) <= group
    engine.close()
    return outputs


@pytest.mark.parametrize(
    "length,chunk,budget", [(1, 8, 1), (7, 2, 1), (7, 3, 4), (17, 8, 9), (11, 32, 4)]
)
def test_grouped_matches_split(length, chunk, budget):
    torch.set_num_threads(1)
    model = tiny_ouro_model()
    reference = execute(model, 1, length, chunk, budget)
    for group in (2, 4):
        assert execute(model, group, length, chunk, budget) == reference


def test_group_does_not_advance_decode_rows():
    with pytest.raises(ValueError, match="mixed decode"):
        SchedulerConfig(mode="mixed", prefill_loop_group_size=2)


def test_invalid_group_rejected():
    with pytest.raises(ValueError, match="group_size"):
        SchedulerConfig(prefill_loop_group_size=3)


def test_last_short_group_and_cancelled_id_reuse():
    model = tiny_ouro_model()
    engine = StageEngine(
        model,
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=8, prefill_chunk_size=8
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
    )
    params = SamplingParams(max_tokens=2, max_loops=4)
    engine.add_request("reused", [1, 2, 3], params)
    for _ in range(4):
        engine.step()  # prelude, R1, R2, R3
    assert engine.scheduler.requests["reused"].prefill_depth(0) == 3
    engine.scheduler.config = replace(
        engine.scheduler.config, prefill_loop_group_size=2
    )
    engine.step()
    assert engine.last_schedule.requests[0].loop_count == 1
    assert engine.scheduler.requests["reused"].phase == RequestPhase.CODA
    engine.abort_request("reused")
    engine.add_request("reused", [4, 5, 6], params)
    engine.step()
    engine.step()  # staged group before cancellation
    engine.abort_request("reused")
    engine.add_request("reused", [9, 8], params)
    for _ in range(100):
        outputs = engine.step()
        if outputs and outputs[-1].finished:
            assert outputs[-1].exit_depths == [4, 4]
            break
    else:
        pytest.fail("Reused request did not finish")
    engine.close()
