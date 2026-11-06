"""CPU regression tests for placement and the native warm-up barrier."""
import pytest
from .hooks import local_placement, separate_schedule, validate_prompt_span


def test_native_warmup_keeps_transfer_owned_readiness():
    for rid in ('deep_primary_serial_warmup', 'recovery_warm_full_P_batch_0',
                'recovery_warm_short_D_geometry_0', 'r-1_q0'):
        assert not local_placement(rid, all_local=True)


def test_fixed_placement_matches_across_local_arms():
    assert [i for i in range(128) if local_placement(f'r0_q{i}')] == list(range(0, 128, 4))


def test_local_qualification_does_not_change_native_warmup():
    assert all(local_placement(f'r0_q{i}', all_local=True) for i in range(32))
    assert local_placement('dlocal_warm_primary')
    assert not local_placement('r-1_q0', all_local=True)


def test_admission_restored_coda_is_not_a_false_progress_stall():
    class Scheduler:
        admitted = False
        def has_ready(self, phase):
            return self.admitted and phase == 'coda'
        def _admit(self):
            self.admitted = True
        def _take(self, phase):
            return phase
    assert separate_schedule(Scheduler(), ('coda','decode'), ('prefill',)) == 'coda'


def test_ready_decode_precedes_prefill_and_new_admission():
    class Scheduler:
        def has_ready(self, phase):
            return phase in ('decode','prefill')
        def _admit(self):
            raise AssertionError('Ready decode must be selected first')
        def _take(self, phase):
            return phase
    assert separate_schedule(Scheduler(), ('coda','decode'), ('prefill',)) == 'decode'


def test_role_scoped_atomic_and_chunk_contracts():
    validate_prompt_span('prefill','paced',0,4096,4096,32)
    validate_prompt_span('decode','paced',32,32,4096,32)
    validate_prompt_span('decode','paced',4096,1,4096,32)
    with pytest.raises(AssertionError):
        validate_prompt_span('prefill','paced',0,32,4096,32)
    with pytest.raises(AssertionError):
        validate_prompt_span('decode','paced',0,64,4096,32)


def test_service_start_excludes_warmup_seed_collisions(tmp_path):
    import json
    from .analyze import stream_starts
    rows=[dict(seed=7,first_model_stream_entry=dict(time=t)) for t in (90.,112.,110.)]
    (tmp_path/'service_device_times_1.json').write_text(json.dumps(dict(requests=rows)))
    assert stream_starts(tmp_path,100.)=={7:110.}
