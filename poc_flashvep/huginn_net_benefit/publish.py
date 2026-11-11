"""Publish only this campaign through a private index. Keep the checkout intact."""
from .common import *


def main():
    configure()
    from ..completion_aware_loop_pd import publish as reused
    for key, value in dict(ROOT=ROOT, SOURCE=SOURCE, BASE=BASE, BRANCH=BRANCH, REPO=REPO).items():
        setattr(reused, key, value)
    reports = REPO/'poc_flashvep/reports'
    reused.DOCUMENTS = [reports/name for name in (
        'huginn_autonomous_adaptive_loop_pd_net_benefit_optimization_spec.md',
        'huginn_autonomous_adaptive_loop_pd_net_benefit_final_summary.md')]
    reused.main()


if __name__ == '__main__':
    main()
