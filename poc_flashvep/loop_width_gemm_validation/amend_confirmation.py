"""Retain the selected arms and add matched-work / strong Token challengers."""
from .common import *
from .prepare_serving import cell


def main():
    initialize();path=ROOT/'state/confirmation_selection.json';plan=json.loads(path.read_text())
    assert 'revision' not in plan
    write(ROOT/'state/confirmation_selection_v1.json',plan)
    jobs=[]
    for cap,arms in ((4,['zero','loop28','token','loop','token1792']),
            (16,['zero','loop28','token1792'])):
        for start,seed in enumerate((2026111111,2026111112,2026111113)):
            order=arms[start%len(arms):]+arms[:start%len(arms)]
            for arm in order:jobs.append(cell('huginn',arm,128,cap,seed,'HELD_OUT_ARRIVAL_CONFIRMATION_SELECTED_ON_N64','v2'))
    for seed in (2026111111,2026111112,2026111113):
        jobs.append(cell('ouro','zero',128,32,seed,'HELD_OUT_ARRIVAL_CONFIRMATION_SELECTED_ON_N64','v2'))
    plan.update(revision=2,jobs=jobs,secondary_challengers=[
        dict(model='huginn',cap=4,arm='loop',reason='One full Loop versus Token64 has equal recurrent rows. Confirm the exact requested geometry.'),
        dict(model='huginn',cap=4,arm='token1792',reason='Best Token TTFT under absolute SLO, despite failing the initial relative-ITL point gate. Retain this stronger throughput challenger.')],
        amendment_scope='Before every N128 result. Original selected arms remain. No successful N128 result selected this amendment.')
    write(path,plan);print('CONFIRMATION_JOBS',len(jobs))


if __name__=='__main__':main()
