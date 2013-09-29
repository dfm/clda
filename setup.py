#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension

setup(
    name="nlp",
    ext_modules=[
        Extension("hw3._viterbi", ["hw3/_viterbi.c"]),
    ],
)
