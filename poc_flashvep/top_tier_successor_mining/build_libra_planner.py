"""Compile the supplied, unmodified Cython planner without importing SGLang.

This avoids loading an old serving stack merely for CPU algorithm parity.
The full pinned SGLang environment is prepared separately.
"""
import argparse
from pathlib import Path
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension,setup


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    module=Extension("prefetch_rebalance_cython",[str(args.source.resolve())],
                     include_dirs=[np.get_include()],
                     define_macros=[("NPY_NO_DEPRECATED_API","NPY_1_7_API_VERSION")],
                     extra_compile_args=["-O3","-march=native","-funroll-loops"])
    setup(name="libra-supplement-planner-audit",ext_modules=cythonize([module],
        build_dir=str(args.out/"cython")),
        script_args=["build_ext","--build-lib",str(args.out),"--build-temp",str(args.out/"temp")])


if __name__=="__main__":main()
