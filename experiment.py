#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from itertools import product
from hw3_runner import main
from multiprocessing import Pool

pool = Pool()


def baseline():
    cmd = "--stupid --greedy --context --thresh {0}"
    thresholds = [1, 5, 10, 50, 100, 1000, 10000]
    cmds = [cmd.format(t).split() for t in thresholds]
    results = pool.map(main, cmds)
    with open("baseline.txt", "w") as f:
        for t, r in zip(thresholds, results):
            args = (map("{0}".format, [t, r[0]])
                    + ["{0:.3f}".format(v*100) for v in r[1:]])
            f.write(" & ".join(args) + "\\\\\n")


def trigram():
    cmd = "--stupid --greedy --thresh {0} --lambda2 {1} --lambda3 {2}"
    thresholds = [2, 5, 10]
    lambda2s = [0.2, 0.3, 0.4]
    lambda3s = [0.5, 0.6, 0.7]
    args = [(t, l2, l3)
            for t, l2, l3 in product(thresholds, lambda2s, lambda3s)
            if l2+l3 < 1.0]
    cmds = [cmd.format(*a).split() for a in args]
    results = pool.map(main, cmds)
    with open("trigram.txt", "w") as f:
        for a, r in zip(args, results):
            cols = (["{0}".format(a[0])] + map("{0:.1f}".format, a[1:])
                    + ["{0:.3f}".format(v*100) for v in r[1:]])
            f.write(" & ".join(cols) + "\\\\\n")


def viterbi():
    cmd = "--stupid --thresh {0} --lambda2 {1} --lambda3 {2}"
    thresholds = [2, 5, 10]
    lambda2s = [0.2, 0.3, 0.4]
    lambda3s = [0.5, 0.6, 0.7]
    args = [(t, l2, l3)
            for t, l2, l3 in product(thresholds, lambda2s, lambda3s)
            if l2+l3 < 1.0]
    cmds = [cmd.format(*a).split() for a in args]
    results = pool.map(main, cmds)
    with open("viterbi.txt", "w") as f:
        for a, r in zip(args, results):
            cols = (["{0}".format(a[0])] + map("{0:.1f}".format, a[1:])
                    + ["{0:.3f}".format(v*100) for v in r[1:]])
            f.write(" & ".join(cols) + "\\\\\n")


def unknown():
    cmd = ("--thresh {0} --lambda2 0.2 --lambda3 0.7 --theta1 {1} "
           "--theta2 {2} --theta3 {3}")
    thresholds = [2, 5, 10]
    theta1s = [0.01, 0.05, 0.1]
    theta2s = [0.3, 0.5, 0.7]
    theta3s = [0.3, 0.5, 0.7]
    args = [(t, t1, t2, t3)
            for t, t1, t2, t3 in product(thresholds, theta1s, theta2s, theta3s)
            if t1+t2+t3 < 1.0]
    cmds = [cmd.format(*a).split() for a in args]
    results = pool.map(main, cmds)
    with open("unknown.txt", "w") as f:
        for a, r in zip(args, results):
            cols = (["{0}".format(a[0])] + map("{0:.1f}".format, a[1:])
                    + ["{0:.3f}".format(v*100) for v in r[1:]])
            f.write(" & ".join(cols) + "\\\\\n")


if __name__ == "__main__":
    # baseline()
    # trigram()
    # viterbi()
    unknown()
