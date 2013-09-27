#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["State", "Trellis"]

from collections import defaultdict

from .data import START, STOP


class Counter(object):

    def __init__(self):
        self.counts = defaultdict(int)
        self.norm = 0

    def __getitem__(self, k):
        return self.counts[k] / self.norm

    def incr(self, k, v=1):
        self.counts[k] += v
        self.norm += v


class State(object):

    def __init__(self, tag1, tag2, ind):
        self.tag1 = tag1
        self.tag2 = tag2
        self.ind = ind

    def __str__(self):
        return "[{0}, {1}, {2}]".format(self.tag1, self.tag2, self.ind)

    def __hash__(self):
        return str(self)

    @classmethod
    def start(cls):
        return cls(START, START, 0)

    @classmethod
    def end(cls, ind):
        return cls(STOP, STOP, ind)

    def next(self, tag):
        return State(self.tag2, tag, self.ind+1)

    def previous(self, tag):
        return State(tag, self.tag1, self.ind-1)


class Trellis(object):

    def __init__(self):
        self.forward_transitions = defaultdict(Counter)
        self.backward_transitions = defaultdict(Counter)


class GreedyDecoder(object):

    def __init__(self):
        pass
