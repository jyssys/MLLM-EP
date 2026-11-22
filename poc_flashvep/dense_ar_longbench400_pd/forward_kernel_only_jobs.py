"""Measure short-input prompt work without API tracing or subsequent decode."""
import argparse
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=int,choices=(0,1));args=parser.parse_args()
    cut=(2,4)[args.case];first=(0,4)[args.case]
    capsule,sha=freeze('short_forward_kernel_only_v1_20261121')
    data=json.loads((ROOT/'data/input0_discovery_exact16.json').read_text())
    suffixes,work=matched_suffixes([r['input_tokens'] for r in data],cut);assert work['exact_match']
    jobs=[];triplets=[]
    for axis in ('Layer','Token'):
        for start in range(3):
            names=[]
            for label in ('Clean1','KernelOnly','Clean2'):
                profile=label=='KernelOnly';extra={}
                if axis=='Token':extra.update(suffix_by_request={'q'+str(i):v for i,v in enumerate(suffixes)},aggregate_work_audit=work)
                name=create(f'Forward1_I0_{axis}{cut}_s{start}_{label}_v1',axis,'input0_discovery_exact16.json',
                    cut=cut,pair=(first,first+1),output=1,fixed=True,controlled=True,serial=True,
                    qualification_gate='C0_full_state_q4_gate.json',warmup_all_prompts=True,
                    source_capsule=capsule,source_capsule_SHA256=sha,
                    kernel_activity=profile,kernel_activity_roles=['prefill','decode'],
                    collector_namespace='kernel_only_collector_v3',activity_copies=True,
                    runtime_api_activity=False,operator_correlation=False,graph_enclosures=False,
                    kernel_activity_seconds=7200,
                    scope='Isolated prompt-forward calibration. Exactly one output token,so no subsequent decode is measured. '
                        'ITL and TPOT are NOT_APPLICABLE. This cannot prove decode safety. '
                        'Warmup uses all benchmark prompt shapes and is not an online serving result. '
                        'Kernel names identify only calibrated categories. Unclassified activity remains explicit.',**extra)
                jobs.append(name);names.append(name)
            triplets.append(dict(axis=axis,cut=cut,start=start,jobs=names))
    plan(f'forward_kernel_only_case{args.case}_v1',jobs)
    write(ROOT/'configs'/f'forward_kernel_only_case{args.case}_observer_v1.json',dict(triplets=triplets,
        evidence='FROZEN_ISOLATED_CALIBRATION_DESIGN',
        observer_rule='Three paired independent triplets. Require the95percent profile/mean-clean CI within[.95,1.05] '
            'for TTFTp95,meanTTFT,P submit-to-ready,D submit-to-first-worker. Require zero dropped or invalid activity. '
            'ITL and TPOT do not exist for one output token. No decode protection claim.',
        attribution='Use exact kernel-name/category associations from earlier call-correlated serving captures. '
            'A name with conflicting categories is unclassified. Report mapped duration and all unknown duration. '
            'Do not infer GEMM M/N/K from an ambiguous kernel name.'))
    print('Frozen18 short-input forward observer starts,cut',cut,flush=True)


if __name__=='__main__':main()
