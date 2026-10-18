"""Freeze a calibration-relative latency proxy before live placement results."""
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    path=ROOT/'artifacts/slo_frozen.json'
    assert not path.exists()
    rows=[r for r in read(ROOT/'artifacts/request_depth_summary.json') if r['split']=='calibration']
    write(path,dict(E2E_SLO_ms=float(2*np.quantile([r['E2E_ms'] for r in rows],.95)),
        definition='Two times the calibration B1 E2E P95. A relative research proxy, not a production SLO.',
        quality_condition='Same selected adaptive runtime. No new benchmark-quality claim for capped discovery.',
        before_live=True))

if __name__=='__main__':main()
