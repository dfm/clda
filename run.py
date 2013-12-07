#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import time
import argparse
import numpy as np
import cPickle as pickle
from multiprocessing import Pool

np.random.seed(1000005)

from ctr.lda import LDA
from ctr.arxiv import ArxivReader

parser = argparse.ArgumentParser(description="Run OVLDA on the arxiv")
parser.add_argument("outdir", help="The results directory")
parser.add_argument("-v", "--vocab", default="data/vocab.txt",
                    help="The vocabulary file")
parser.add_argument("-k", "--ntopics", default=100, type=int,
                    help="The number of topics")
parser.add_argument("-b", "--batch", default=1024, type=int,
                    help="The number of documents in a mini-batch")
parser.add_argument("--tau", default=1024, type=float)
parser.add_argument("--kappa", default=0.5, type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    try:
        os.makedirs(args.outdir)
    except os.error:
        print("Output directory exists. Overwriting")

    reader = ArxivReader("data/abstracts.db")
    reader.load_vocab(args.vocab, skip=100, nvocab=8000)
    with open(os.path.join(args.outdir, "vocab.txt"), "w") as f:
        f.write("\n".join(reader.vocab_list))
    pickle.dump(reader, open(os.path.join(args.outdir, "reader.pkl"), "wb"),
                -1)

    # Load a validation set.
    validation = reader.validation(1024)
    nvalid = sum([len(s) for s in validation])

    # Set up the model.
    pool = Pool(8)
    model = LDA(args.ntopics, len(reader.vocab_list), 0.01, 0.01,
                tau=args.tau, kappa=args.kappa)
    p = model.elbo(validation, pool=pool)

    # Run EM.
    fn = os.path.join(args.outdir, "model.{0:04d}.pkl")
    outfn = os.path.join(args.outdir, "output.log")
    open(outfn, "w").close()
    tot = 0.0
    nxt = 2.0
    ndocs = 8e5
    strt = time.time()
    for i, (n, lam) in enumerate(model.em(reader, ndocs=ndocs, pool=pool,
                                          batch=args.batch)):
        if np.log10(tot+time.time()-strt) > nxt:
            tot += time.time() - strt
            p = np.exp(-model.elbo(validation, pool=pool, ndocs=ndocs)/nvalid)
            print(i, tot, p)
            open(outfn, "a").write("{0} {1} {2}\n".format(i*args.batch, tot,
                                                          p))
            pickle.dump(model, open(fn.format(i), "wb"), -1)
            strt = time.time()
            nxt += 0.1
