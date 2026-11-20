"""Test the observer before interpreting quiescent windows."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def main():
    declaration=ROOT/'configs/frozen_BI0_quiescent_window_design_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=32,QPS=2.35,seed=2026112033,
            order=['clean_before','profile','clean_after'],copies=True,seconds=40,
            observer_limit=.05,scope='A completed CUDA event and empty D scheduler certify each observed quiescent left boundary. '
            'The future endpoint remains unknown online. No offload execution or fit guarantee.'))
    names={}
    for label in ('clean_before','profile','clean_after'):
        name=f'Window_Native_N32_q2.35_{label}_BI0_v1';names[label]=name
        serving(name,'Native','qualification32.json',32,2.35,2026112033,
            phase='QUIESCENT_WINDOW_OBSERVER_CONTROL',quiescent_witness=True,
            kernel_activity=label=='profile',activity_copies=True,kernel_activity_seconds=40)
        analyze(name)
    result=summarize_triplet(2.35,names)
    write(ROOT/'analysis/BI0_quiescent_window_observer_gate.json',result)
    write(ROOT/'state/quiescent_window_complete.json',dict(time=time.time(),complete=True,
        observer_passed=result['observer_gate_passed']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/quiescent_window_hold.json',dict(time=time.time(),error=repr(error)));raise
