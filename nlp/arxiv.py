#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import glob
import gzip
import operator
import numpy as np
from collections import defaultdict


class ArxivReader(object):

    def __init__(self, basepath):
        # Construct the list of files.
        self.files = glob.glob(os.path.join(basepath, "*", "*.txt.gz"))
        self.nfiles = len(self.files)
        assert self.nfiles, "Couldn't find any data files"
        self.validation_set = []

    def read_random_file(self):
        with gzip.open(self.files[np.random.randint(self.nfiles)]) as f:
            return [line.split("\t") for line in f]

    def iter_docs(self):
        while True:
            for doc in self.read_random_file():
                yield doc

    def validation(self, n):
        docs = []
        for doc in self.iter_docs():
            if doc[0] in self.validation_set:
                continue
            words = self.parse_document(doc)
            if not len(words):
                continue
            self.validation_set.append(doc[0])
            docs.append(words)
            if len(docs) >= n:
                break
        return docs

    def parse_document(self, doc):
        return [self.vocab[w.lower()]
                for w in doc[2].split()+doc[3].split()
                if w.lower() in self.vocab]

    def generate_vocab(self, ndocs, skip=0, nvocab=50000):
        vocab = defaultdict(int)
        for count, doc in enumerate(self.iter_docs()):
            for w in doc[2].split()+doc[3].split():
                vocab[w.lower()] += 1
            if count >= ndocs:
                break
        return [w for w, v in sorted(vocab.iteritems(),
                                     key=operator.itemgetter(1),
                                     reverse=True)[skip:skip+nvocab]]

    def load_vocab(self, fn, skip=0, nvocab=None):
        self.vocab = {}
        self.vocab_list = []
        with open(fn, "r") as f:
            for count, w in enumerate(f):
                if count < skip:
                    continue
                self.vocab_list.append(w.strip())
                self.vocab[w.strip()] = len(self.vocab_list) - 1
                if nvocab is not None and len(self.vocab_list) >= nvocab:
                    break

    def __iter__(self):
        for doc in self.iter_docs():
            if doc[0] in self.validation_set:
                continue
            words = self.parse_document(doc)
            if len(words):
                yield words


if __name__ == "__main__":
    reader = ArxivReader("/export/bbq1/dfm/research/data.arxiv.io/data")
    reader.load_vocab("vocab.txt")
    # open("vocab.txt", "w").write("\n".join(reader.generate_vocab(500000)))
