#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["read_tagged_sentences", "get_vocabulary"]

START = "<S>"
STOP = "</S>"


class BoundedList(object):

    def __init__(self, items, start=START, stop=STOP):
        self.start = start
        self.stop = stop
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            return self.items[i]
        except IndexError:
            if i < 0:
                return self.start
            return self.stop


class TaggedSentence(object):

    def __init__(self, words, tags):
        self.words = BoundedList(words)
        self.tags = BoundedList(tags)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return (self.words[i], self.tags[i])

    def __str__(self):
        return " ".join(["{0}_{1}".format(self.words[i], self.tags[i])
                         for i in range(len(self.words))])

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if (self.words[i] != other.words[i]
                    or self.tags[i] != other.tags[i]):
                return False
        return True


def read_tagged_sentences(path):
    sentences = [[]]
    with open(path) as f:
        for line in f:
            if line == "\n":
                sentences.append([])
                continue

            columns = line.strip().split("\t")
            word, tag = columns if len(columns) > 1 else (columns[0], None)
            sentences[-1].append((word, tag))

    return [TaggedSentence(*zip(*s)) for s in sentences if len(s)]


def get_vocabulary(sentences):
    vocab = set([])
    for s in sentences:
        vocab.update(set(s.words.items))
    return vocab


if __name__ == "__main__":
    training = read_tagged_sentences("data/en-wsj-train.pos")
    print(get_vocabulary(training))
