"""Publish scoped evidence without modifying the user's Git index."""
from .common import *


def main():
    configure()
    from ..completion_aware_loop_pd import publish as reused
    for key,value in dict(ROOT=ROOT,SOURCE=SOURCE,BASE=BASE,BRANCH=BRANCH,REPO=REPO).items():
        setattr(reused,key,value)
    reused.DOCUMENTS=[REPO/'poc_flashvep/reports'/name for name in (
        'ouro_huginn_loop_width_vs_token_chunk_gemm_validation_spec.md',
        'ouro_huginn_loop_width_vs_token_chunk_gemm_validation_final_summary.md')]
    original=reused.git
    def scoped_git(*args,**kwargs):
        command=list(args)
        if command and command[0]=='commit-tree' and '-m' in command:
            position=command.index('-m')+1
            metadata='Record verified' in command[position]
            command[position]=('Record verified Loop-width GEMM publication' if metadata else
                'Validate Loop width against tuned Token chunks in exact PD serving')
        return original(*command,**kwargs)
    reused.git=scoped_git
    try:reused.main()
    finally:reused.git=original


if __name__=='__main__':main()
