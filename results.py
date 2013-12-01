#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import argparse
import numpy as np

from ctr.lda import dirichlet_expectation
from ctr.arxiv import ArxivReader

parser = argparse.ArgumentParser(description="Show OVLDA results")
parser.add_argument("outdir", help="The results directory")
parser.add_argument("lam", help="The path to the results file")

if __name__ == "__main__":
    args = parser.parse_args()
    reader = ArxivReader("/export/bbq1/dfm/research/data.arxiv.io/data")
    reader.load_vocab(os.path.join(args.outdir, "vocab.txt"))

    lam = np.loadtxt(args.lam)
    lnbeta = dirichlet_expectation(lam)
    for i, topics in enumerate(lnbeta):
        inds = np.argsort(topics)
        print("Topic {0:3d}: ".format(i) +
              " ".join([reader.vocab_list[i] for i in inds[-10:]]))
