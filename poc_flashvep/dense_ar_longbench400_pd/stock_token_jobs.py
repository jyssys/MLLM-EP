"""Test a stronger stock-model token runtime before claiming C3 improvements."""
from .experiment import create
from .next_jobs import plan


def main():
    base=dict(pair=(0,1),correctness=True,output=16,fixed=True,serial=True,
        execution_family='INSTALLED_DEFAULT_COMPILE',wall_timeout=1200,serving_timeout=900)
    jobs=[create('Q8_StockFull36_TokenTeacher8_N8_v1','Native','qualification8.json',
        reference_single=True,token_teacher_cut=8,state_capture=True,axis_state_capture=False,
        numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',**base),
        create('Q8_StockToken8_N8_FullState_v1','Token','qualification8.json',cut=8,
            controlled=True,token_stock_model=True,axis_state_capture=True,
            axis_capture_hidden_last_only=True,axis_hash_workers=8,**base),
        create('Q8_StockToken8_C3_Natural32_v1','Token','continuous_qualification32_natural.json',cut=8,
            pair=(0,1),correctness=True,token_stock_model=True,
            execution_family='INSTALLED_DEFAULT_COMPILE',qps=8,seed=2026112231,
            axis_state_capture=True,axis_capture_hidden_last_only=True,axis_hash_workers=8,
            continuous_source_credit_limit=0,wall_timeout=1800,serving_timeout=1500)]
    plan('stock_model_token_qualification_v1',jobs)
    print('Frozen default-compile stock-model token qualification.',flush=True)


if __name__=='__main__':main()
