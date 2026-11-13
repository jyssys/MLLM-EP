"""Freeze empirical uncertainty from previously viewed starts, not fresh tests."""
import math,statistics
from .common import *
from .analyze import percentile


def main():
    path=ROOT/'configs/revised_cost_model_v1.json'
    if path.exists():print('COST_ALREADY_FROZEN');return
    training=json.loads((ROOT/'analysis/prior_viewed_uncertainty_training.json').read_text())
    source=training['source_errors_ms'];completion=training['completion_errors_ms'];drain=training['known_drain_errors_ms']
    model=json.loads((ROOT/'configs/calibrated_cost_model_previous.json').read_text())
    model['uncertainty']=dict(source_bias_ms=statistics.mean(source),
        source_absolute_p95_ms=math.ceil(percentile([abs(x) for x in source],.95)),
        completion_upper_margin_ms=10*math.ceil(max(0.,percentile(completion,.95))/10),
        known_drain_underprediction_p95_ms=10*math.ceil(max(0.,percentile(drain,.95))/10),
        training_SHA256=digest(ROOT/'analysis/prior_viewed_uncertainty_training.json'),
        source_samples=len(source),completion_samples=len(completion),known_drain_samples=len(drain),
        future_fresh_confirmation_information=False,
        scope='Empirical small-sample upper margins. Source bias and completion margin are separate. No WCET or production guarantee.')
    model.update(campaign_frozen_epoch=time.time(),previous_LUT_SHA256=digest(ROOT/'configs/calibrated_cost_model_previous.json'),
        changed_GPU_math=False,scope='Reuse exact Native phase costs. Validate new online central predictions and upper coverage separately. Roofline is a structural lower bound only.')
    write(path,model);print(json.dumps(model['uncertainty'],indent=2))


if __name__=='__main__':main()
