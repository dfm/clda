#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import numpy as np
import cPickle as pickle

from ctr.lda import dirichlet_expectation

parser = argparse.ArgumentParser(description="Show OVLDA results")
parser.add_argument("reader", help="The results directory")
parser.add_argument("model", help="The path to the results file")

if __name__ == "__main__":
    args = parser.parse_args()
    reader = pickle.load(open(args.reader))
    model = pickle.load(open(args.model))
    lnbeta = dirichlet_expectation(model.lam)
    for i, topics in enumerate(lnbeta):
        inds = np.argsort(topics)
        print("Topic {0:3d}: ".format(i) +
              " ".join([reader.vocab_list[i] for i in inds[-15:]]))
