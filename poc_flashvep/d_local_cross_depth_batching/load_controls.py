"""Measure native arrival-rate controls without expanding a failed candidate."""
from .prepare import *
from .run import execute


def main():
    initialize()
    rates=dict(ouro=dict(low=1.23460875,heavy=6.356),huginn=dict(low=.37903355455,heavy=1.971))
    for load in ('heavy','low'):
        for model in ('ouro','huginn'):
            cfg=configuration(model,rate=rates[model][load])
            cfg.update(name=f'load_{model}_native_{load}_n128',load=load)
            save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
