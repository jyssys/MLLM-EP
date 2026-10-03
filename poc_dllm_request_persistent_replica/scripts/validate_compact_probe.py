"""Full inherited physical validator on a losslessly reconstructed long probe."""
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd
from poc_dllm_request_persistent_replica.contract import ROOT,new_json
from poc_dllm_request_persistent_replica.scripts.pack_evidence import verify
from poc_dllm_ep_comm.scripts.analyze import request


def main():
    name='coding_00_b32'
    with TemporaryDirectory(prefix='dllm-persistent-compact-validation-') as temp:
        dest=Path(temp)/name
        verify(ROOT/'raw/compact'/name,dest)
        restored,_,_,_=request(dest)
        original=pd.read_csv(ROOT/'results/long_requests.csv').set_index('request').loc[name]
        for key in ('B_actual','B_live_only','B_one_shot','nfe','output_tokens','physical_buffer_validation_checks'):
            assert restored[key]==original[key],key
        assert restored['semantic_parity']
    new_json(ROOT/'results/compact_probe_validation.json',dict(status='PASS',request=name,
        restored_named_arrays_exact=True,restored_JSON_bytes_exact=True,
        full_original_physical_ownership_receive_parity_validator=True,accounting_and_NFE_identical=True))
    print('COMPACT_PHYSICAL_PROBE_ROUNDTRIP_PASS',flush=True)


if __name__=='__main__':main()
