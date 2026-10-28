"""CPU checks for report accounting. No GPU inference occurs here."""
import pytest
from .critical_path_analysis import PHASES,partition_statistics,fixed_timeline_oracle

def record(elapsed,values):
 return dict(phase_order_valid=True,harness_E2E_ms=elapsed,
  **{name+'_ms':value for name,value in zip(PHASES,values)})

def test_host_partition_has_one_explicit_denominator():
 rows=[record(100,[10,10,10,10,10,40,10]),record(200,[0,0,0,0,0,200,0])]
 result=partition_statistics(rows)
 assert sum(r['share_of_summed_request_E2E_percent'] for r in result)==pytest.approx(100)
 assert result[0]['share_of_summed_request_E2E_percent']==pytest.approx(100/30)
 assert result[0]['mean_request_share_percent']==pytest.approx(5)

def test_host_partition_rejects_a_missing_interval():
 with pytest.raises(ValueError,match='does not equal'):
  partition_statistics([record(100,[1,1,1,1,1,1,1])])

def test_fixed_timeline_oracle_does_not_remove_prefill_or_decode():
 rows=[record(100,[10,10,10,20,10,30,10])]
 result=fixed_timeline_oracle(rows)
 assert result['frozen_timeline_p95_E2E_ms']==80
 assert result['frozen_timeline_p95_ratio']==1.25
 assert result['evidence']=='ORACLE_FROZEN_TIMELINE_NOT_MEASURED_SPEEDUP'

def test_invalid_order_is_not_silently_clamped():
 row=record(100,[0,0,0,-10,0,110,0]);row['phase_order_valid']=False
 assert partition_statistics([row])==[]
 assert fixed_timeline_oracle([row]) is None
