#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
    name="nlp",
    ext_modules=[
        Extension("hw3._viterbi", ["hw3/_viterbi.c"]),
        Extension("hw3._maxent", ["hw3/_maxent.c"],
                  include_dirs=get_numpy_include_dirs(),
                  libraries=["lbfgs"],
                  library_dirs=["/usr/local/lib"],
                  extra_link_args=["-Wl,-rpath",
                                   "-Wl,/usr/local/lib"]),
    ],
)
