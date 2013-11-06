#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

from math import log
from collections import defaultdict

from nltk.corpus.reader import BracketParseCorpusReader


class Counter(defaultdict):

    def normalize(self):
        norm = log(sum(self.values()))
        [self.__setitem__(k, log(v) - norm) for k, v in self.items()]


class Parser(object):

    def __init__(self, grammar, lexicon):
        self.grammar = grammar
        self.lexicon = lexicon

    def generate_parse_tree(self, sentence):
        n = len(sentence)
        score = [[defaultdict(float) for j in range(n-k)] for k in range(n)]
        back = [[defaultdict(float) for j in range(n-k)] for k in range(n)]

        # Apply lexical rules.
        for i, w in enumerate(sentence):
            for t in self.lexicon.tags:
                s = self.lexicon.score(w, t)
                if s is not None:
                    score[i][0][t] = s
                    back[i][0][t] = (i, 0, w)

        # Initial unary pass.
        for i, w in enumerate(sentence):
            self.update_unaries(sentence, i, 0, score, back)

        print(score)

    def update_unaries(self, sentence, start, end, score, back):
        dist = score[start][end]
        added = True
        while added:
            added = False
            for parent, rules in self.grammar.unaries.items():
                for child, prob in rules.items():
                    if child in dist:
                        p = dist[child] + prob
                        if parent not in dist or p > dist[parent]:
                            added = True
                            dist[parent] = p
                            back[start][end][parent] = (start, end, child)

        # assert 0

        # for i, w in enumerate(sentence):
        #     for t in self.lexicon.tags:
        #         tableau[i][0][t] = (self.lexicon.score(w, t), [w])

        # for span in range(1, n):
        #     for begin in range(0, n-span):
        #         end = begin + span
        #         for split in range(begin+1, end):
        #             left = tableau[begin][split]
        #             right = tableau[split][end]
        #             for top, rules in self.grammar.binaries.items():
        #                 for rule, prob in rules.items():
        #                     B, C = rule.rhs()
        #                     if B not in left and C not in right:
        #                         continue
        #                     p = left[B][0] * right[C][0] * prob
        #                     if p > tableau[begin][end][top][0]:
        #                         tableau[begin][end][top] = (p, rule)

        # return tableau


class MiniGrammar(object):

    def __init__(self, rules):
        self.unaries = defaultdict(lambda: Counter(float))
        self.binaries = defaultdict(lambda: Counter(float))

        for l, r, p in rules:
            if len(r.split()) == 1:
                self.unaries[l][r] += p
            else:
                self.binaries[l][r] += p

        [self.unaries[k].normalize() for k in self.unaries]
        [self.binaries[k].normalize() for k in self.binaries]


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
                if len(children) == 1:
                    self.unaries[node][p] += 1
                else:
                    self.binaries[node][p] += 1

        [self.unaries[k].normalize() for k in self.unaries]
        [self.binaries[k].normalize() for k in self.binaries]


class Lexicon(object):

    def __init__(self, tagged_words):
        raise NotImplementedError("Convert to log-probs")

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
        raise NotImplementedError("Convert to log-probs and return None")

        p_tag = self.tags[tag]
        if p_tag == 0:
            return -float("inf")
        c_word = self.words[word]
        c_tag_and_word = self.words_to_tags[word][tag]
        if c_word < 10:
            c_word += 1
            c_tag_and_word += self.tag_types[tag]
        p_word = (1 + c_word) / (1 + self.total)
        p_tag_given_word = c_tag_and_word / c_word
        return p_tag_given_word / p_tag * p_word

    def best_tag(self, word):
        tags = self.tags.keys()
        scores = [(t, self.score(word, t)) for t in tags]
        return max(scores, key=lambda e: e[1])[0]


class MiniLexicon(Lexicon):

    def __init__(self, rules):
        self.tags = defaultdict(lambda: Counter(float))
        for t, w, p in rules:
            self.tags[t][w] += p
        [self.tags[k].normalize() for k in self.tags]

    def score(self, word, tag):
        t = self.tags[tag]
        return t[word] if word in t else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data")
    parser.add_argument("--maxTrain", default=999, type=int)
    parser.add_argument("--horizontal", default=None, type=int)
    parser.add_argument("--vertical", default=0, type=int)
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    if args.mini:
        grammar = MiniGrammar([
            ("S", "NP VP", 0.9),
            ("S", "VP", 0.1),
            ("VP", "V NP", 0.5),
            ("VP", "V", 0.1),
            ("VP", "V @VP_V", 0.3),
            ("VP", "V PP", 0.1),
            ("@VP_V", "NP PP", 1.0),
            ("NP", "NP NP", 0.1),
            ("NP", "NP PP", 0.2),
            ("NP", "N", 0.7),
            ("PP", "P NP", 1.0),
        ])
        lexicon = MiniLexicon([
            ("N", "people", 0.5),
            ("N", "fish", 0.2),
            ("N", "tanks", 0.2),
            ("N", "rods", 0.1),
            ("V", "people", 0.1),
            ("V", "fish", 0.6),
            ("V", "tanks", 0.3),
            ("P", "with", 1.0),
        ])

    else:
        # Load the training data.
        print("Loading training trees")
        train_corpus = BracketParseCorpusReader(args.data,
                                                "en-wsj-train.1.mrg")
        train_trees = train_corpus.parsed_sents()

        print("Building grammar")
        grammar = Grammar(train_corpus, max_train=args.maxTrain,
                          horizontal=args.horizontal, vertical=args.vertical)

        print("Building lexicon")
        lexicon = Lexicon(train_corpus.tagged_words())

    # Set up and train the parser.
    parser = Parser(grammar, lexicon)

    if args.mini:
        tree = parser.generate_parse_tree(["fish", "people", "fish", "tanks"])
        print(tree)
