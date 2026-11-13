"""Reuse private-index publication without changing the user's index."""
from . import common

DOCUMENTS=['dense_ar_decode_safe_elastic_layer_pd_poc_spec.md',
    'dense_ar_decode_safe_elastic_layer_pd_final_summary.md',
    'dense_ar_continuous_colocated_correctness_and_baseline.md',
    'dense_ar_dynamic_layer_cut_decision_analysis.md',
    'dense_ar_layer_cost_model_and_deadline_validation.md',
    'dense_ar_four_method_qps_sweep_results.md',
    'dense_ar_static_dynamic_matched_migration_comparison.md']


def main():
    from ..dense_ar_dynamic_layer_pd import publish as implementation
    for name in ('ROOT','REPO','SOURCE','BASE','BRANCH','preserve','preservation','digest','write','git'):
        setattr(implementation,name,getattr(common,name))
    implementation.DOCUMENTS=DOCUMENTS
    implementation.main()


if __name__=='__main__':main()
