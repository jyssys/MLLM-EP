"""Test one lower operating rate without changing the frozen method."""
import fcntl
import json
import time
from pathlib import Path

from poc_flashvep.dense_ar_slo_layer_pd.common import (
    ROOT, REPO, digest, write, save, preserve, resources, QUIET_LOCK,
)
from poc_flashvep.dense_ar_slo_layer_pd.confirmation import SELECTION
from poc_flashvep.dense_ar_slo_layer_pd.sustainable import duration_config, assess
from poc_flashvep.dense_ar_slo_layer_pd.run import execute
from poc_flashvep.dense_ar_slo_layer_pd.qualification import against_unseen
from poc_flashvep.dense_ar_slo_layer_pd.analyze import metrics

PROTOCOL = ROOT / 'configs/frozen_method_lower_capacity_protocol.json'
COMPLETE = ROOT / 'analysis/frozen_method_lower_capacity_complete.json'
SEEDS = (2026111516, 2026111517, 2026111518)


def unchanged():
    frozen = json.loads((ROOT / 'state/heldout_runtime_freeze.json').read_text())
    assert all(digest(REPO / name) == value for name, value in frozen['hashes'].items())
    preserve()


def declare():
    start = json.loads((ROOT / 'state/progress.json').read_text())['start_epoch']
    value = dict(
        method='Dynamic', rate=4.4, seeds=list(SEEDS),
        measurement_seconds=600., warm_seconds=60., follow_seconds=90.,
        selection_SHA256=digest(SELECTION), script_SHA256=digest(Path(__file__)),
        frozen_runtime_SHA256=digest(ROOT / 'state/heldout_runtime_freeze.json'),
        stop_measurements_epoch=start + 21 * 3600,
        reason='The frozen4.8 point failed all3 independent starts. A failed point cannot define sustainable capacity. Test only the existing lower4.4 candidate with the same method and fresh arrival seeds.',
        decision_rule='Do not change a method parameter, cost table, SLO or dataset. Run3 starts only if every preceding lower-point start passes. Stop after any failed screen or runtime failure. Do not test another lower rate. Retain every attempted run.',
        scope='A prospective supplementary rate check, not part of the original frozen confirmation selection. The prompt pool remains the same, already viewed confirmation pool. The arrival seeds are new. This checks rate robustness, not generalization to new prompt IDs.',
    )
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        assert all(old[key] == val for key, val in value.items())
    else:
        write(PROTOCOL, dict(time=time.time(), **value))
    return value


def original_controller_active():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            words = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'poc_flashvep.dense_ar_slo_layer_pd.confirmation_campaign' in words:
                return True
        except OSError:
            pass
    return False


def main():
    protocol = declare()
    while original_controller_active():
        assert time.time() < protocol['stop_measurements_epoch'], 'No budget remains for the lower-point check'
        time.sleep(10)
    assert (ROOT / 'analysis/independent_confirmation_complete.json').exists()
    assert not (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    selected = json.loads(SELECTION.read_text())['methods']['Dynamic']
    names = []
    reason = 'ALL_THREE_LOWER_POINT_STARTS_COMPLETED'
    try:
        for seed in SEEDS:
            if time.time() + 1050 > protocol['stop_measurements_epoch']:
                reason = 'NOT_RUN_REMAINING_STARTS_BUDGET'
                break
            unchanged()
            name = f'heldout_backoff_Dynamic_O16_q4.4_s{seed}_W600_v1'
            options = dict(selected['options'])
            options.update(
                duration_prompt_pool='unseen_confirmation.json',
                intern_immutable_prompt_pool=True,
                phase='FROZEN_METHOD_SUPPLEMENTARY_LOWER_RATE_CONFIRMATION',
                selected_method_label='Dynamic',
                selection_SHA256=digest(SELECTION),
                supplementary_rate_protocol_SHA256=digest(PROTOCOL),
            )
            cfg = duration_config(name, selected['arm'], 4.4, cap=16, seed=seed,
                warm=60., measure=600., follow=90., **options)
            assert execute(save(cfg), restore=False)
            assert against_unseen(name)['passed']
            value = assess(name)
            write(ROOT / 'analysis' / (name + '_metrics.json'), metrics(name))
            names.append(name)
            print('FROZEN_METHOD_BACKOFF', name,
                value['SLO_regimes']['Decode_priority'], flush=True)
            write(COMPLETE, dict(time=time.time(), names=names,
                status='IN_PROGRESS', protocol_SHA256=digest(PROTOCOL)))
            if not value['SLO_regimes']['Decode_priority']['qualified']:
                reason = 'STOP_AFTER_FAILED_LOWER_POINT_SCREEN'
                break
        unchanged()
        write(COMPLETE, dict(time=time.time(), names=names, status='COMPLETE',
            reason=reason, protocol_SHA256=digest(PROTOCOL),
            evidence='MEASURED' if names else 'NOT_RUN',
            scope='All attempted lower-rate starts remain in the evidence. The original failed4.8 confirmation is unchanged. A lower-rate result does not establish Native-relative noninferiority at4.4 without a same-rate Native reference.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
