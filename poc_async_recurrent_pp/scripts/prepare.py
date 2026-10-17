"""Freeze source documents and metadata before GPU experiments."""
import urllib.request
import datetime
from poc_async_recurrent_pp.contract import *


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    assert not (ROOT/'reports/spec.md').exists()
    source = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/async_recurrent_pp_poc_spec.md')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/spec.md').write_bytes(source.read_bytes())
    metadata = dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_original_requested='poc_async_recurrent_pp/reports/spec.md',
        spec_found=str(source), spec_sha256=sha(source), model_revision=REVISION,
        primary_dtype='BF16', R=4, stages=[[0,6],[6,12],[12,18],[18,24]],
        attention='SDPA', exact_prefill_all_arms=True, seed=20261017,
        initial_output_cap=2048, expanded_output_cap=8192,
        quality_stop='Stop before distributed work if both W1 and W0 materially fail pilot quality.',
        split_rule='Calibration: GSM8K train. Pilot: hashed GSM8K test and MATH-500 IDs, frozen before inference.')
    sources = ROOT/'sources'
    sources.mkdir(exist_ok=True)
    upstream = {}
    for repo in ['czg1225/AsyncDiff', 'kschwethelm/continuous-depth-batching']:
        info = json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/HEAD'))
        commit = info['sha']
        content = fetch('https://raw.githubusercontent.com/'+repo+'/'+commit+'/README.md')
        name = repo.split('/')[-1]
        (sources/(name+'_README.md')).write_bytes(content)
        upstream[repo] = dict(commit=commit, readme_sha256=sha(sources/(name+'_README.md')))
    metadata['upstream'] = upstream
    write(ROOT/'outputs/manifests/contract.json', metadata)
    print(metadata)


if __name__ == '__main__':
    main()
