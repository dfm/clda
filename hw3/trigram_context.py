#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []


def extract_sentence_contexts(sentence):
    contexts = set([])
    for i in range(len(sentence)+2):
        contexts.update(LocalTrigramContext(sentence.words, i,
                                            sentence.tags[i-2],
                                            sentence.tags[i-1],
                                            sentence.tags[i]))
    return contexts


def extract_contexts(sentences):
    return set([context
                for contexts in map(extract_sentence_contexts, sentences)
                for context in contexts])


class LocalTrigramContext(object):

    def __init__(self, words, position, tag1, tag2, tag=None):
        self.words = words
        self.position = position
        self.tag1 = tag1
        self.tag2 = tag2
        self.tag = tag

    def __str__(self):
        if self.tag is None:
            return "[{0}, {1}, {2}]".format(self.tag1, self.tag2,
                                            self.words[self.position])
        return "[{0}, {1}, {2}_{3}]".format(self.tag1, self.tag2,
                                            self.words[self.position],
                                            self.tag)

    def __hash__(self):
        return str(self)
