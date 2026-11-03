"""Collect a separate D trace without calling it simultaneous P/D evidence."""
from .common import *
from .run import execute
from .analysis_lane import quiet_analysis


def main():
    while time.time() < LONG_START_CUTOFF:
        paths = [ROOT/'state'/(n+'.json') for n in ('qualified_probe_curve', 'huginn_shorter', 'long_ouro_confirmation')]
        if all(p.exists() and json.loads(p.read_text()).get('stage') in
               ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'NO_D_SLO_CANDIDATE') for p in paths):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    records = []
    for model, source in (('ouro', 'qualified_ouro_s1_nsys128'), ('huginn', 'window_huginn_s1_nsys384')):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        cfg.update(name=f'decode_timeline_{model}', capture_role='decode')
        write(ROOT/'configs'/(cfg['name']+'.json'), cfg)
        write(ROOT/'state/decode_timeline.json', dict(stage='PROFILE', name=cfg['name'], time=time.time()))
        row = dict(model=model, name=cfg['name'], scope='Separate D-only capture. Do not combine this start with a P-only profile as simultaneous device evidence.')
        if execute(cfg['name']):
            with quiet_analysis(cfg['name']+'_analysis'):
                row['profile'] = analyze(cfg['name'], cfg['matched_clean'])
        else:
            row['status'] = 'PROFILE_FAILED'
        records.append(row)
        write(ROOT/'state/decode_timeline_results.json', records)
    write(ROOT/'state/decode_timeline.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/decode_timeline.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
