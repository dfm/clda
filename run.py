#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import time
import argparse
import numpy as np
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

if __name__ == "__main__":
    args = parser.parse_args()
    try:
        os.makedirs(args.outdir)
    except os.error:
        print("Output directory exists. Overwriting")

    reader = ArxivReader("data/abstracts.db")
    reader.load_vocab(args.vocab, skip=100, nvocab=20000)
    with open(os.path.join(args.outdir, "vocab.txt"), "w") as f:
        f.write("\n".join(reader.vocab_list))

    # Load a validation set.
    validation = reader.validation(1024)
    nvalid = sum([len(s) for s in validation])

    # Set up the model.
    pool = Pool()
    model = LDA(args.ntopics, len(reader.vocab_list), 0.01, 0.01)
    p = model.elbo(validation, pool=pool)

    # Run EM.
    fn = os.path.join(args.outdir, "lambda.{0:04d}.txt")
    outfn = os.path.join(args.outdir, "output.log")
    open(outfn, "w").close()
    tot = 0.0
    ndocs = 8e5
    strt = time.time()
    for i, (n, lam) in enumerate(model.em(reader, ndocs=ndocs, pool=pool)):
        if i % 10 == 0:
            tot += time.time() - strt
            p = np.exp(-model.elbo(validation, pool=pool, ndocs=ndocs)/nvalid)
            print(tot, p)
            open(outfn, "a").write("{0} {1}\n".format(tot, p))
            np.savetxt(fn.format(i), lam)
            strt = time.time()
