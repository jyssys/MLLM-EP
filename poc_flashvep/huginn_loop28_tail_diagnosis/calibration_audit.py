"""Trace the literal to its matching Host calibration, not a GPU WCET."""
from .common import *
from .analyze import latency


def main():
    initialize()
    name='selection_group32_seed2026110940'
    folder=OLD_ROOT/'raw'/name
    measured=latency(folder)['ITL_ms']['p99']
    literal=93.10616207122803
    old_forecasts=[]
    for name in ('forecast8b_output16_loop_seed2026110940','forecast8_output16_token_seed2026110940',
            'forecast8_output4_loop_seed2026110940','forecast8_output4_token_seed2026110940'):
        files=list((OLD_ROOT/'raw'/name).glob('predictive_checkpoint_preparation_*.json'))
        rows=[r for p in files for r in json.loads(p.read_text())['records']]
        old_forecasts.append(dict(name=name,preparation_records=len(rows),
            known_cohort_request_ids_logged=False,advertisement_observed_time_logged=False,
            exact_drain_prediction_error='NOT_IDENTIFIED',
            scope='P preparation receipts contain a duration forecast and remaining-output count. They omit the original advertisement time and exact D cohort IDs. Do not score them against all future arrivals or infer a target cohort from a delivery batch.'))
    write(ROOT/'analysis/forecast_calibration_provenance.json',dict(
        literal_ms=literal,source='poc_flashvep/huginn_net_benefit/forecast_cells.py',
        consumer='poc_flashvep/huginn_net_benefit/predictive_prepare.py',
        matching_control='selection_group32_seed2026110940',matching_host_ITL_p99_ms=measured,
        absolute_numeric_difference_ms=abs(literal-measured),
        source_configuration_sha256=digest(OLD_ROOT/'configs/selection_group32_seed2026110940.json'),
        source_requests_sha256=digest(folder/'requests.jsonl'),
        formula='max_remaining_outputs * max(configured_cycle_floor, recent_observed_interval_p95)',
        used_for='Checkpoint preparation only. Actual offload admission remains a separate current-state decision.',
        provenance_limit='The number is a source-code literal. Its near-exact numeric match to this Host ITL p99 identifies the calibration datum. The source does not contain a machine-verifiable derivation manifest.',
        excluded_interpretations=['Not a CUDA kernel time','Not a physical GPU idle duration','Not a WCET guarantee','Not a measured sustainable service cycle'],
        historical_records=old_forecasts))
    print('CALIBRATION',measured,literal,abs(measured-literal))


if __name__=='__main__':main()
