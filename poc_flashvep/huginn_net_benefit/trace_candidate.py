"""Move expensive collector lifecycle calls outside the measured cohort."""
from .common import *
from .prepare import clone


def main():
    initialize()
    name = 'diagnostic_zero_graph_steady_seed2026110931'
    cfg = clone('control_huginn_zero_seed2026110931', name,
        phase='GRAPH_ENCLOSURE_OVERHEAD_QUALIFICATION_NOT_KERNEL_UNION',
        local_nsys=True, nsight_graph_enclosures=True,
        nsight_start_before_measurement=True, clean_confirmation=False,
        startup_timeout=240, timeout=240)
    save(cfg)
    print(name)


if __name__ == '__main__':
    main()
