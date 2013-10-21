#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from hw4 import run
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


def heuristic_test():
    args = "--model heuristic -n {0} -o heuristic"
    ns = [500, 1000, 5000, 10000, 100000, 500000]
    results = [run(args.format(n).split()) for n in ns]
    with open("heuristic.txt", "w") as f:
        for n, r in zip(ns, results):
            f.write("{0:d} & {1:.3f} & {2:.3f} & {3:.3f} \\\\\n"
                    .format(n, *r))

    results = np.array(results)
    fig, axes = pl.subplots(2, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.01, hspace=0.08)
    axes[0].plot(np.log10(ns), results[:, 0], "k")
    axes[0].plot(np.log10(ns), results[:, 1], "--k")
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_ylabel("precision, recall")
    axes[1].plot(np.log10(ns), results[:, 2], "k")
    axes[1].yaxis.set_major_locator(MaxNLocator(3))
    axes[1].set_ylabel("AER")
    axes[1].set_xlabel("$\log_{10}\,N_\mathrm{train}$")
    fig.savefig("heuristic.pdf")


def model1_test():
    args = "--model model1 -n {0} -o model1_{0} --niter 10"
    ns = [500, 1000, 5000, 10000, 100000, 500000]
    results = [run(args.format(n).split()) for n in ns]
    with open("model1.txt", "w") as f:
        for n, r in zip(ns, results):
            f.write("{0:d} & {1:.3f} & {2:.3f} & {3:.3f} \\\\\n"
                    .format(n, *r))

    results = np.array(results)
    fig, axes = pl.subplots(2, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.01, hspace=0.08)
    axes[0].plot(np.log10(ns), results[:, 0], "k")
    axes[0].plot(np.log10(ns), results[:, 1], "--k")
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_ylabel("precision, recall")
    axes[1].plot(np.log10(ns), results[:, 2], "k")
    axes[1].yaxis.set_major_locator(MaxNLocator(3))
    axes[1].set_ylabel("AER")
    axes[1].set_xlabel("$\log_{10}\,N_\mathrm{train}$")
    fig.savefig("model1.pdf")


def model2_test():
    args = "--model model2 -n {0} -o model2_{0} --niter 10"
    ns = [500, 1000, 5000, 10000, 100000, 200000]
    results = [run(args.format(n).split()) for n in ns]
    with open("model2.txt", "w") as f:
        for n, r in zip(ns, results):
            f.write("{0:d} & {1:.3f} & {2:.3f} & {3:.3f} \\\\\n"
                    .format(n, *r))

    results = np.array(results)
    fig, axes = pl.subplots(2, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.01, hspace=0.08)
    axes[0].plot(np.log10(ns), results[:, 0], "k")
    axes[0].plot(np.log10(ns), results[:, 1], "--k")
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_ylabel("precision, recall")
    axes[1].plot(np.log10(ns), results[:, 2], "k")
    axes[1].yaxis.set_major_locator(MaxNLocator(3))
    axes[1].set_ylabel("AER")
    axes[1].set_xlabel("$\log_{10}\,N_\mathrm{train}$")
    fig.savefig("model2.pdf")


if __name__ == "__main__":
    model2_test()
    # model1_test()
    # heuristic_test()
