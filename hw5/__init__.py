#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

from collections import defaultdict

import nltk
from nltk.treetransforms import chomsky_normal_form
from nltk.corpus.reader import BracketParseCorpusReader


class Counter(defaultdict):

    def normalize(self):
        norm = sum(self.values())
        [self.__setitem__(k, v / norm) for k, v in self.items()]


class Parser(object):

    def __init__(self, corpus, max_train=999, horizontal=None, vertical=0):
        self.corpus = corpus

        print("Building grammar")
        self.grammar = Grammar(self.corpus, max_train=max_train,
                               horizontal=horizontal, vertical=vertical)

        print("Building lexicon")
        self.lexicon = Lexicon(self.corpus.tagged_words())

    def generate_parse_tree(self, sentence):
        pass


class Grammar(object):

    def __init__(self, corpus, max_train=999, horizontal=None, vertical=0):
        self.unaries = defaultdict(lambda: Counter(int))
        self.binaries = defaultdict(lambda: Counter(int))

        for i, tree in enumerate(corpus.parsed_sents()):
            if len(tree.leaves()) > max_train:
                continue
            tree.chomsky_normal_form(horzMarkov=horizontal,
                                     vertMarkov=vertical)
            for p in tree.productions():
                if p.is_lexical():
                    continue
                node = p.lhs()
                children = p.rhs()
                assert len(children) in [1, 2], "{0} {1}".format(p, i)
                self.symbols[node] += 1
                if len(children) == 1:
                    self.unaries[node][p] += 1
                else:
                    self.binaries[node][p] += 1


class Lexicon(object):

    def __init__(self, tagged_words):
        self.tag_types = Counter(float)
        self.tags = Counter(float)
        self.words = Counter(float)
        self.words_to_tags = defaultdict(lambda: Counter(float))
        self.total = 0

        # Count the words, tags and pairs.
        for word, tag in tagged_words:
            # Unknown word model.
            if word in self.words:
                self.tag_types[tag] += 1
            self.tags[tag] += 1
            self.words[word] += 1
            self.words_to_tags[word][tag] += 1
            self.total += 1

        self.tag_types.normalize()
        self.tags.normalize()

    def score(self, word, tag):
        p_tag = self.tags[tag]
        c_word = self.words[word]
        c_tag_and_word = self.words_to_tags[word][tag]
        if c_word < 10:
            c_word += 1
            c_tag_and_word += self.tag_types[tag]
        p_word = (1 + c_word) / (1 + self.total)
        p_tag_given_word = c_tag_and_word / c_word
        return p_tag_given_word / p_tag * p_word


if __name__ == "__main__":
    pass
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data", default="data")
    # parser.add_argument("--maxTrain", default=999, type=int)
    # parser.add_argument("--horizontal", default=None, type=int)
    # parser.add_argument("--vertical", default=0, type=int)
    # args = parser.parse_args()

    # # Load the training data.
    # print("Loading training trees")
    # train_corpus = BracketParseCorpusReader(args.data, "en-wsj-train.1.mrg")
    # train_trees = train_corpus.parsed_sents()

    # # Set up and train the parser.
    # parser = Parser(train_corpus, max_train=args.maxTrain,
    #                 horizontal=args.horizontal, vertical=args.vertical)
