"""Record task-metric dependencies without changing the scoring environment."""
from email.parser import Parser
import importlib.util
import json
import sys
import time
from .common import ROOT, write, digest


def main():
    parent = ROOT / 'tools/quality_deps'
    distributions = []
    for folder in sorted(parent.glob('*.dist-info')):
        metadata = Parser().parsestr((folder / 'METADATA').read_text())
        distributions.append(dict(name=metadata['Name'], version=metadata['Version'],
            license=metadata.get('License'), requires_dist=metadata.get_all('Requires-Dist', []),
            metadata_SHA256=digest(folder / 'METADATA')))
    assert {d['name'].lower():d['version'] for d in distributions} == {
        'fuzzywuzzy':'0.18.0', 'jieba':'0.42.1', 'rouge':'1.0.1', 'six':'1.17.0'}
    files = [dict(path=str(p.relative_to(parent)), bytes=p.stat().st_size, SHA256=digest(p))
             for p in sorted(parent.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    sys.path.insert(0, str(parent))
    from fuzzywuzzy import fuzz
    backend = fuzz.SequenceMatcher.__module__
    assert backend == 'difflib', backend
    path = ROOT / 'references/quality_dependency_audit_v1.json'
    value = dict(time=time.time(), distributions=distributions, files=files,
        official_metrics_SHA256=digest(ROOT / 'references/official_longbench_metrics.py'),
        fuzzywuzzy_sequence_backend=backend,
        evidence='CPU_REPRODUCTION_ENVIRONMENT_AUDIT',
        scope='Read-only audit during final serving. This is not an initial-environment timestamp proof. '
              'The local quality dependency tree is not redistributed as a self-contained runtime. '
              'Install the exact versions in a separate environment and compare hashes before reproducing scores. '
              'Do not add python-Levenshtein silently. Frozen task-quality results and raw answers remain unchanged.')
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ('distributions','files','official_metrics_SHA256','fuzzywuzzy_sequence_backend'):
            assert previous[key] == value[key], key
    else:
        write(path, value)
    print('TASK_QUALITY_DEPENDENCIES_VERIFIED', len(distributions), len(files), backend, flush=True)


if __name__ == '__main__':
    main()
