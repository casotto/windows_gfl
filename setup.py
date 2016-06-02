from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
from os.path import join

extensions = [Extension("gfl.solver", [ "gfl/solver.pyx",
                                        "gfl/cpp/src/graph_fl.c",
                                        "gfl/cpp/src/tf_dp.c",
                                        "gfl/cpp/src/csparse.c"])]

# Use cythonize on the extension object.
setup(name="gfl",
      packages=["gfl"],
      ext_modules = cythonize(extensions),
      include_dirs=[np.get_include(), join('gfl','cpp','include')])
