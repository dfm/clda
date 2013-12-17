#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["_function_wrapper"]


class _function_wrapper(object):

    def __init__(self, target, attr, *args, **kwargs):
        self.target = target
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def __call__(self, v):
        return getattr(self.target, self.attr)(v, *self.args, **self.kwargs)
