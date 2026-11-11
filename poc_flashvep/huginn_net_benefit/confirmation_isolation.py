"""Retain a source-scan overlap and predeclare one clean replacement."""
from .common import *


def resolve(name, replacements):
    return replacements.get(name, name)


def main():
    initialize()
    original = 'primaryconfirmv2_loopfour_seed2026110952'
    replacement = original+'_isolationrepeat'
    folder = ROOT/'raw'/original
    result_exists = (folder/'dispatcher_result.json').exists()
    cfg = json.loads((ROOT/'configs'/(original+'.json')).read_text())
    cfg['name'] = replacement
    cfg['isolation_replaces'] = original
    save(cfg)
    write(ROOT/'state/primaryconfirmv2_isolation_replacements.json', dict(
        time=time.time(), replacements={original:replacement},
        reason='A source search took more than ten seconds while the original measured cohort ran. CPU and I/O interference are possible, not measured. Repeat as an isolation precaution.',
        command="rg -n 'def linear_by_row_tiles_padded|B_NUMERICAL_VARIANT|ROW_TILE' poc_flashvep | head -n 45",
        conservative_overlap_wall_UTC=['2026-11-11 02:37:00','2026-11-11 02:37:52'],
        selection_rule='Use the replacement regardless of its result. Retain and report the original as an isolation sensitivity observation. Do not select the better result.',
        original_result_not_inspected_at_declaration=True,
        original_dispatcher_receipt_existed_at_declaration=result_exists,
        initial_prepare_guard='The first preparation required an unfinished cell. It failed because the cell had already finished. The original performance values were not inspected before this repeat decision. This is an isolation-based exclusion, not a pre-result statistical selection.'))
    print(replacement)


if __name__ == '__main__':
    main()
