# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

from itertools import izip, imap

from glob import iglob
from nltk import sent_tokenize, word_tokenize


class GlobReader(object):

    def __init__(self, pattern):
        self.pattern = pattern
        self.validation_set = []

    def parse_document(self, doc):
        return [w for l in imap(word_tokenize, sent_tokenize(doc)) for w in l]

    def iter_docs(self):
        while True:
            try:
                for fn in iglob(self.pattern):
                    with open(fn, "r") as f:
                        doc = self.parse_document(f.read())
                    yield fn, doc
            except StopIteration:
                pass

    def __iter__(self):
        for did, doc in self.iter_docs():
            if did in self.validation_set:
                continue
            words = self.parse_document(doc)
            if len(words):
                yield words
