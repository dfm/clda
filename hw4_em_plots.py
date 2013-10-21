#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


def em_plot(root):
    data = np.loadtxt(os.path.join(root, "em.txt"))
    alpha = len(data[0]) == 5

    if alpha:
        fig, axes = pl.subplots(3, 1, figsize=(6, 6))
    else:
        fig, axes = pl.subplots(2, 1, figsize=(6, 6))

    fig.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.01, hspace=0.08)

    niter = data[:, 0] + 1
    if alpha:
        results = data[:, 2:]
    else:
        results = data[:, 1:]

    axes[0].plot(niter, results[:, 0], "k")
    axes[0].plot(niter, results[:, 1], "--k")
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel("precision, recall")

    axes[1].plot(niter, results[:, 2], "k")
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    axes[1].set_ylabel("AER")

    if alpha:
        axes[1].set_xticklabels([])

        axes[2].plot(niter, data[:, 1], "k")
        axes[2].yaxis.set_major_locator(MaxNLocator(4))
        axes[2].set_ylabel(r"$\alpha$")

        axes[2].set_xlabel("$\log_{10}\,N_\mathrm{iter}$")
    else:
        axes[1].set_xlabel("$\log_{10}\,N_\mathrm{iter}$")

    fig.savefig(os.path.join(root, "convergence.pdf"))


if __name__ == "__main__":
    import sys
    em_plot(sys.argv[1])
