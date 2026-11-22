"""Append explicit hypotheses,tests,and evidence limits without changing raw data."""
import json
import time
from .common import ROOT,write


def record(identifier,observation,hypotheses,test,result,next_step,evidence):
    path=ROOT/'analysis/research_decisions.json'
    rows=json.loads(path.read_text())['rows'] if path.exists() else []
    assert identifier not in {r['id'] for r in rows}
    rows.append(dict(id=identifier,time=time.time(),observation=observation,hypotheses=hypotheses,
        discriminating_test=test,result=result,next_step=next_step,evidence=evidence))
    write(path,dict(rows=rows))


def initial():
    record('R001','The archived C3 adapter rejects a batch with partial-prefill and decode requests.',
        ['A batch-wide intermediate tensor is applied to rows that need different starting layers.',
         'The first request action is reused for all requests.',
         'Hidden/KV lifetimes do not follow each request progress.'],
        'Execute the actual archived prepared_inputs body with one partial request and one decode request.',
        'The exact archived assertion fails. The fixture is CPU-only. The new implementation uses per-row layer intervals and fresh subset attention metadata.',
        'Require real N1,N8,N32 and Natural400 GPU coverage,teacher and lifetime gates.','CPU_MINIMAL_REPRODUCTION')
    record('R002','All eight natural-reference workers fail before generation because ninja is absent from PATH.',
        ['The selected Python environment contains ninja,but the inherited executable search path does not.'],
        'Execute the environment ninja binary. Repeat initialization with that environment bin directory first in PATH.',
        'The binary reports version1.13.2.git.kitware.jobserver-pipe-1. All eight v2 workers complete2165 natural outputs.',
        'Preserve failed v1 logs and use the completed v2 reference.','MEASURED_EXECUTION')
    record('R003','The declared candidate universe has only3 long-input/output25–49 and16 long-input/output50–99 requests.',
        ['The full16-cell balance is infeasible in this natural-output candidate population.'],
        'Generate every declared feasible original prompt before selection and count all16 cells.',
        'Select400 original requests with100 per input bin. The long-input selected row is52/29/3/16. No serving metric selects requests.',
        'Report the declared-universe limit and the population shift. Audit additional source coverage separately without changing frozen requests.','MEASURED_NATURAL_OUTPUT_REFERENCE')
    record('R004','Token and Hybrid differ from a full-query single-GPU teacher at many BF16 KV tensors.',
        ['Different legal query geometry changes BI0 numerics.',
         'The imported prefix KV is wrong.',
         'A causal suffix dependency is missing.'],
        'Use an original full36 single-GPU token-boundary teacher and an independent local three-wave hybrid teacher.',
        'N1 Token and Hybrid each match their geometry teacher in all36 full valid KV hashes,greedy16 tokens,and last hidden. N8 imported-prefix hashes and coverage pass. This does not qualify Natural400.',
        'Extend to mixed/variable requests and task-quality gates.','CORRECTNESS_ONLY')
    record('R005','Prior destination profiling ends with a collector API error.',
        ['An empty completed buffer causes GetNextRecord to reject validSize0.',
         'Warmup/capture records differ from steady serving records.',
         'The callback completion counter becomes visible before its buffer record is stored.',
         'Graph/kernel record parsing or a dropped-record query fails.'],
        'Call the installed CUPTI GetNextRecord API with a non-null empty buffer. Export per-buffer parser diagnostics in the new collector.',
        'The empty-buffer API returns CUPTI_ERROR_MAX_LIMIT_REACHED12. The first hypothesis is falsified. Do not suppress that error by assumption.',
        'Test a short destination capture after warmup. Preserve parser return codes and validate graph-kernel coverage.','CPU_API_DIAGNOSTIC')


if __name__=='__main__':initial()
