#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, Extension

ext = Extension("ctr._cf", ["ctr/_cf.c"],
                include_dirs=[numpy.get_include(), "ctr"],
                libraries=["blas", "lapack"])

setup(
    name="str",
    ext_modules=[ext],
)
