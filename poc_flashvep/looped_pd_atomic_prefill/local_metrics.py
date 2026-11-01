"""Enable the existing StageEngine work counters in a local reference worker."""
from .common import ROOT,os


def prepare(cfg):
    os.environ['ABC_RUN_DIR']=str(ROOT/'raw'/cfg['name'])
