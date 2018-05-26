from distutils.core import setup
from Cython.Build import cythonize

setup(name="cymatrix", ext_modules=cythonize('cymatrix.pyx'),)
