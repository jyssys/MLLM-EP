"""Reuse scoped alternate-index publication for this campaign only."""
import argparse
from .common import *

FINAL=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_gated_loop_continuation_final_summary.md'
SPEC=FINAL.with_name('ouro_huginn_adaptive_idle_gated_loop_continuation_poc_spec.md')


def main(action,push=False):
    initialize()
    from ..d_local_cross_depth_batching import publish as inherited
    for key in ('ROOT','SOURCE','BASE','BRANCH','FINAL','SPEC','initialize'):
        setattr(inherited,key,globals()[key])
    from ..looped_pd_autonomous_headroom import publish_campaign
    original=publish_campaign.git
    def scoped_git(*args,**kwargs):
        if args and args[0]=='commit-tree':
            args=tuple(value.replace('D-local cross-depth batching PoC','adaptive idle-gated continuation PoC')
                .replace('verified D-local publication receipt','verified adaptive idle-gated publication receipt')
                if isinstance(value,str) else value for value in args)
        return original(*args,**kwargs)
    publish_campaign.git=scoped_git
    try:inherited.main(action,push)
    finally:publish_campaign.git=original


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('archive','publish'))
    parser.add_argument('--push',action='store_true');args=parser.parse_args();main(args.action,args.push)
