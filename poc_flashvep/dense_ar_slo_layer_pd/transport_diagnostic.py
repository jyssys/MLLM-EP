"""Inspect a measured transport hypothesis before changing primary runs."""
from .common import *
from .configuration import base
from .frontier import run


def main():
    jobs=[base('diag_Colocated_S1_B8192_N64_O16_host_v1','Colocated',chunk=8192,token_budget=8192,
        D_slots=1,use_instrumented_worker=True,control_plane_timing=True,clean_timing=False,
        phase='HOST_INGRESS_DIAGNOSTIC_NOT_PRIMARY'),
        base('diag_Native_N32_descriptor_order_observe_v1','Native',count=32,rate=4.8,
            descriptor_order_probe=True,sort_descriptor_pairs=False,clean_timing=False,
            use_instrumented_worker=True,control_plane_timing=True,
            phase='PAIRED_DESCRIPTOR_ORDER_OBSERVATION_NOT_PRIMARY')]
    run(jobs)


if __name__=='__main__':main()
