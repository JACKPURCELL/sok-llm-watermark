from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='LevenshteinModule',
    ext_modules=cythonize("levenshtein.pyx"),
    include_dirs=[numpy.get_include()]
)
