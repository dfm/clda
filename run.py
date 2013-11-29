#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from nlp.lda import LDA
from nlp.arxiv import ArxivReader

if __name__ == "__main__":
    reader = ArxivReader("/export/bbq1/dfm/research/data.arxiv.io/data")
    reader.load_vocab("vocab.txt", nvocab=2000)

    # Load a validation set.
    validation = reader.validation(1024)

    # Set up the model.
    model = LDA(50, len(reader.vocab), 0.01, 0.01)

    ndocs = 8e5
    for elbo in model.em(reader, ndocs=ndocs):
        print(elbo, model.elbo(validation))
