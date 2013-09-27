#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MostFrequentTagScorer"]


class MostFrequentTagScorer(object):

    def __init__(self, restrict=False):
        self.restrict = restrict
