#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, Extension

setup(
    name="nlp",
    ext_modules=[
        Extension("hw5._cky", ["hw5/_cky.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
