# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GlobReader"]

import string
from glob import iglob
from collections import Counter
from itertools import imap, ifilter

from nltk.corpus import stopwords as swc
from nltk import sent_tokenize, word_tokenize

stopwords = set(swc.words("english") + ["http", "https"])


def clean_token(w, stripchars=string.punctuation + "1234567890"):
    w = w.lower().strip(stripchars)
    return u"" if w in stopwords else w


class GlobReader(object):

    def __init__(self, pattern):
        self.pattern = pattern
        self.validation_set = []

    def parse_document(self, doc, min_length=3, resolve=False):
        tokens = (w for l in imap(word_tokenize, sent_tokenize(doc))
                  for w in l)
        f = ifilter(lambda t: len(t) >= min_length, imap(clean_token, tokens))
        if resolve:
            f = ifilter(lambda w: w is not None, imap(self.vocab.get, f))
        return f

    def iter_docs(self, resolve=False):
        while True:
            try:
                for fn in iglob(self.pattern):
                    with open(fn, "r") as f:
                        try:
                            doc = self.parse_document(f.read()
                                                      .decode("utf-8"),
                                                      resolve=resolve)
                        except UnicodeDecodeError:
                            continue
                    yield fn, doc
            except StopIteration:
                pass

    def __iter__(self):
        for did, doc in self.iter_docs(resolve=True):
            if did in self.validation_set:
                continue
            words = list(doc)
            if len(words):
                yield words

    def validation(self, n):
        docs = []
        for did, words in self.iter_docs(resolve=True):
            words = list(words)
            if did in self.validation_set or not len(words):
                continue
            self.validation_set.append(words)
            docs.append(words)
            if len(docs) >= n:
                break
        return docs

    def generate_vocab(self, N, total=None, save_to_file=None):
        vocab = Counter()
        for n, (did, doc) in enumerate(self.iter_docs()):
            vocab.update(doc)
            if total is not None and (n+1) % 1000 == 0:
                vocab = Counter(dict(vocab.most_common(2*total)))
            if n+1 >= N:
                break
        if total is None:
            vocab = vocab.most_common()
        else:
            vocab = vocab.most_common(total)

        if save_to_file is not None:
            print("Saving vocab to: {0}".format(save_to_file))
            with open(save_to_file, "w") as f:
                for w, c in vocab:
                    f.write(u"{0} {1}\n".format(w, c).encode("utf-8"))

        return vocab

    def load_vocab(self, fn, skip=0, nvocab=None):
        self.vocab = {}
        self.vocab_list = []
        with open(fn, "r") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                w, _ = line.decode("utf-8").split()
                self.vocab_list.append(w.strip())
                self.vocab[w.strip()] = len(self.vocab_list) - 1
                if nvocab is not None and len(self.vocab_list) >= nvocab:
                    break
