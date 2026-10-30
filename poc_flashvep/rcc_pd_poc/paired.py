"""Independent-start uncertainty. No token-level pseudoreplication occurs here."""
import csv
import itertools
import json
import math
import statistics
from .common import ROOT, write
from .analyze import percentile


def main():
    rows = [r for r in csv.DictReader((ROOT/'full_results.csv').open()) if r['name'].startswith('quiet')]
    output = []
    for model in ('ouro','huginn'):
        by_key = {(r['arm'],int(r['name'].rsplit('_',1)[1])):r for r in rows if r['model'] == model}
        for reference, candidate in (('A_sep','A_native_mixed'), ('PD_G1','A_sep'), ('PD_G1','A_native_mixed')):
            repeats = [i for i in range(3) if (reference,i) in by_key and (candidate,i) in by_key]
            for metric in ('makespan_ms','mean_TTFT_ms','mean_TPOT_ms','mean_E2E_ms'):
                ratios = [float(by_key[(reference,i)][metric])/float(by_key[(candidate,i)][metric]) for i in repeats]
                if not ratios:
                    continue
                log_mean = statistics.mean(math.log(v) for v in ratios)
                draws = [math.exp(statistics.mean(math.log(ratios[i]) for i in indices))
                         for indices in itertools.product(range(len(ratios)),repeat=len(ratios))]
                output.append(dict(model=model, reference=reference, candidate=candidate, metric=metric,
                                   independent_starts=len(repeats), requests_per_start=8,
                                   speedup_geomean=math.exp(log_mean), CI95_low=percentile(draws,.025) if len(repeats)>=3 else None,
                                   CI95_high=percentile(draws,.975) if len(repeats)>=3 else None, paired_start_ratios=ratios,
                                   evidence='MEASURED', uncertainty='Paired independent-start bootstrap. Three starts give a coarse interval, not a serving-capacity CI.'))
    write(ROOT/'paired_start_comparisons.json', output)
    if output:
        with (ROOT/'paired_comparisons.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(output)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
