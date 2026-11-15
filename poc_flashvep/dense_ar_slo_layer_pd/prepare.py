"""Keep previously viewed data and fixtures separate from future confirmation."""
from .common import *


def main():
    initialize()
    for name in ('revised_cost_model_v1.json','revised_cost_model_v2.json'):
        target=ROOT/'configs'/name
        if not target.exists():write(target,json.loads((PREVIOUS/'configs'/name).read_text()))
    write(ROOT/'data/discovery_provenance.json',dict(base=BASE,
        viewed_data=True,source=str(PREVIOUS/'data/discovery.json'),
        prompt_SHA256=digest(ROOT/'data/discovery.json'),
        scope='Previously viewed discovery data only. Fresh confirmation must exclude all prior document IDs and prefixes.'))
    write(ROOT/'state/initial_cpu_fixture_erratum.json',dict(
        symptom='Initial copied CPU tests required v1 lookup fixtures. The bootstrap copied only v3.',
        correction='Copy unchanged v1/v2 fixtures into the new namespace before any GPU job.',
        evidence='FAILED_CPU_SETUP_NOT_GPU_MEASUREMENT'))
    print('DISCOVERY_AND_FIXTURES_READY')


if __name__=='__main__':main()
