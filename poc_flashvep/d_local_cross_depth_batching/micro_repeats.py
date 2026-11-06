"""Repeat the matched readiness control from independent model starts."""
from .prepare import *
from .run import execute


def main():
    initialize()
    for start in (2,3):
        for model in ('ouro','huginn'):
            cfg=configuration(model)
            cfg.update(name=f'opportunity_{model}_graph_start{start}',worker='operator_micro',micro_opportunity=True)
            save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
