"""Record the exact inherited parser and symbolic-scoring dependencies."""
import importlib.metadata
import sys
from poc_decode_first_depth_aware_dp.contract import *


def main():
    output=ROOT/'artifacts/scoring_environment.json'
    assert not output.exists()
    dependencies=Path('/home/esjung/external/looped-exact-depth-chunk-assets/cpu_deps')
    sys.path.insert(0,str(dependencies))
    names=['math-verify','sympy','latex2sympy2-extended','antlr4-python3-runtime','numpy']
    versions={name:importlib.metadata.version(name) for name in names}
    sources=[PROJECT/'poc_async_recurrent_pp/scoring.py',PROJECT/'poc_looped_exact_depth_chunk_pipeline/scoring.py']
    write(output,dict(versions=versions,dependency_path=str(dependencies),
        parser_sources={str(p.relative_to(PROJECT)):sha(p) for p in sources},
        interpretation='One identical parser for both policies. A valid extracted answer may count even in a truncated output.'))
    print(versions)


if __name__=='__main__':main()
