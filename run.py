#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from multiprocessing import Pool

from nlp.lda import LDA
from nlp.arxiv import ArxivReader

if __name__ == "__main__":
    reader = ArxivReader("/export/bbq1/dfm/research/data.arxiv.io/data")
    reader.load_vocab("vocab.txt", nvocab=2000)

    # Load a validation set.
    validation = reader.validation(1024)
    nvalid = sum([len(s) for s in validation])

    # Set up the model.
    pool = Pool()
    model = LDA(50, len(reader.vocab), 0.01, 0.01)

    # Run EM.
    fn = "lambda.{0:04d}.txt"
    for i, (n, lam) in enumerate(model.em(reader, ndocs=8e5, pool=pool)):
        if i % 10 == 0:
            validation_bound = model.elbo(validation, pool=pool)
            print(np.exp(-validation_bound / nvalid))
            np.savetxt(fn.format(i), lam)
