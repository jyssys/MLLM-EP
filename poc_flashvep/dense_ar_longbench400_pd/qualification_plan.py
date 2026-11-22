"""Freeze full-state C0 teachers and partition qualifications before timing."""
from .common import ROOT,write
from .experiment import create


def main():
    groups=[[],[],[]];all_jobs=[]
    def add(name,method='Native',cut=0,teacher=False,**extra):
        lane=len(all_jobs)%3;pair=(2*lane,2*lane+1)
        options=dict(pair=pair,qps=20,output=16,fixed=True,correctness=True,serial=True)
        if teacher:
            options.update(reference_single=True,continuous_source_credit_limit=0,
                axis_state_capture=False,state_capture=True,numerical_probe=True,numerical_probe_sync=True,kv_layout='HND')
        else:options['controlled']=True
        options.update(extra)
        create(name,method,'qualification8.json',cut=cut,**options)
        groups[lane].append(name);all_jobs.append(name)
    add('Q4_Full36_Teacher_N8',teacher=True)
    for cut in (2,4,8,12):
        add(f'Q4_Layer{cut}_N8','Layer',cut)
        add(f'Q4_Token{cut}_N8','Token',cut)
        add(f'Q4_TokenTeacher{cut}_N8',teacher=True,token_teacher_cut=cut)
    add('Q4_Hybrid4_f0125_Aligned_N8','Hybrid',4,hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
    add('Q4_HybridBoundaryTeacher_f0125_N8',teacher=True,token_teacher_cut=4,teacher_suffix_fraction=.125)
    add('Q4_Hybrid0_f0125_TokenBoundary_N8','Hybrid',0,hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
    add('Q4_Hybrid4_f0_LayerBoundary_N8','Hybrid',4,hybrid_suffix_fraction=0.,partition_aligned_chunks=True)
    for i,jobs in enumerate(groups):write(ROOT/'jobs'/f'qualification_q4_lane{i}.json',dict(jobs=jobs,stop_on_failure=True))
    write(ROOT/'configs/full_state_qualification_q4_frozen.json',dict(jobs=all_jobs,
        strict_gate='All36 full valid prompt KV hashes,last prompt hidden,greedy16 tokens,query/layer coverage and transfer lifetimes.',
        geometry='C0 actual1/1/1. Teachers use original single-GPU full36 layers. Prefix boundaries reproduce legal query shapes. '
            'Aligned hybrid0/s andm/0 must reproduce pure-axis boundaries. Natural/mixed C3 needs a separate gate.',
        evidence='CORRECTNESS_ONLY'))
    print('QUALIFICATION_Q4_JOBS',len(all_jobs))


if __name__=='__main__':main()
