"""Check full400 stress and same-adapter mode0 controls before performance."""
from .experiment import create
from .next_jobs import plan
from .freeze_capsule import freeze


def main():
    capsule,sha=freeze('stock_token_followup_qualification_v1_20261121')
    common=dict(correctness=True,token_stock_model=True,axis_state_capture=True,
        axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
        source_capsule=capsule,source_capsule_SHA256=sha)
    jobs=[create('Q14_StockToken8_C3_Natural400_GPUdigest_v1','Token','longbench_balanced400_v1.json',cut=8,
        pair=(6,7),qps=2.8,seed=2026112201,execution_family='INSTALLED_DEFAULT_COMPILE',
        gpu_state_digest=True,continuous_source_credit_limit=0,wall_timeout=3600,serving_timeout=3300,**common)]
    plan('stock_token_full400_qualification_v1',jobs)
    jobs=[]
    for cut in (2,8):
        jobs.append(create(f'Q15_StockMode0_Token{cut}_N8_v1','Token','qualification8.json',cut=cut,
            pair=(0,1),output=16,fixed=True,controlled=True,serial=True,
            execution_family='MATCHED_NO_COMPILE',axis_hash_workers=8,
            wall_timeout=1800,serving_timeout=1500,**common))
    plan('stock_token_mode0_control_qualification_v1',jobs)
    print('Frozen full400 stock-token stress and mode0 N8 controls.',flush=True)


if __name__=='__main__':main()
