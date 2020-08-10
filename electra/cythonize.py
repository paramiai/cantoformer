
import sys

sys.argv = [sys.argv[0]] + ['build_ext','--inplace']

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize('*.pyx', 
                          compiler_directives={
                              'boundscheck': False, 
                              'wraparound':False, 
                              'initializedcheck': False,
                              'infer_types': True
                          }, annotate=True),
    include_dirs=[numpy.get_include()]
)    
'''
python cythonize.py
'''