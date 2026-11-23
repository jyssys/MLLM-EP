"""Hold the same resource lease with an explicit predecessor-guard identity."""
import runpy
import sys
from .common import bind

if __name__=='__main__':
    bind();name=sys.argv[1];sys.argv=[name]
    runpy.run_module('poc_flashvep.dense_ar_longbench400_pd.'+name,run_name='__main__')
