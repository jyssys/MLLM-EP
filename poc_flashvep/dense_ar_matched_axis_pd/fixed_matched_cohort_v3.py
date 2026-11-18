"""Keep the Token PrefixChannel around the registered cohort channel."""
import subprocess
import sys
from . import fixed_matched_cohort_v2 as previous_version


def preserve_token_client(cfg,role,worker,atomic):
    if role=='decode' and cfg.get('token_axis'):
        atomic.Client=worker.Client


def coordinator_main():
    previous=subprocess.Popen
    class Popen(previous):
        def __init__(self,args,*a,**kw):
            if isinstance(args,(list,tuple)) and '-m' in args:
                args=list(args);index=args.index('-m')+1
                if args[index]=='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v2':
                    args[index]='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v3'
            super().__init__(args,*a,**kw)
    subprocess.Popen=Popen
    return previous_version.coordinator_main()


def main():
    if '--role' not in sys.argv:
        return coordinator_main()
    from ..dense_ar_slo_layer_pd import native
    previous=native.stage
    def stage(cfg,role,address):
        previous_version.install(cfg,role)
        from ..dense_ar_slo_layer_pd import worker
        from ..looped_pd_atomic_prefill import ar_worker
        preserve_token_client(cfg,role,worker,ar_worker)
        return previous(cfg,role,address)
    native.stage=stage
    native.main()


if __name__=='__main__':
    main()
