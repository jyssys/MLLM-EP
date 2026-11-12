"""Verify source tests and report paths after GPU work finishes."""
import ast,io,json,re,unittest
from .common import *


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    names=['protocol','hidden_slots','environment','physical_analysis','token_protocol','async_ingress','selection_names']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(
        'poc_flashvep.dense_ar_layerwise_pd.test_'+name) for name in names)
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    parsed=[]
    for path in sorted(SOURCE.glob('*.py')):
        ast.parse(path.read_text(),filename=str(path));parsed.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    documents=[REPO/'poc_flashvep/reports'/name for name in (
        'dense_ar_prefill_heavy_layerwise_pd_feasibility_spec.md',
        'dense_ar_layerwise_pd_prior_work_runtime_review.md',
        'dense_ar_prefill_heavy_layerwise_pd_feasibility_final_summary.md')]
    bad=[]
    for path in documents:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            if not (path.parent/target.split('#')[0]).exists():bad.append(dict(document=path.name,target=target))
    write(ROOT/'state/final_verification.json',dict(time=time.time(),tests_run=result.testsRun,
        tests_passed=result.wasSuccessful(),test_output=output.getvalue(),source_AST_checks=parsed,
        missing_local_document_links=bad,preservation=preservation(),
        passed=result.wasSuccessful() and not bad,
        scope='CPU tests and local artifact links. GPU state gates, output parity, metrology qualification, and performance verdicts remain separate.'))
    print(output.getvalue());assert result.wasSuccessful() and not bad,bad


if __name__=='__main__':main()
