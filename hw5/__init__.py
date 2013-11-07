#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
from math import exp, log
from collections import defaultdict

import nltk

from . import _cky


class Counter(defaultdict):

    def __init__(self):
        super(Counter, self).__init__(float)

    def normalize(self, norm=None):
        if norm is None:
            norm = log(sum(self.values()))
        else:
            norm = log(norm)
        [self.__setitem__(k, log(v) - norm) for k, v in self.items()]


class Parser(object):

    def __init__(self, grammar, lexicon):
        self.grammar = grammar
        self.lexicon = lexicon

    def generate_parse_tree(self, sentence, root_tag="TOP"):
        n = len(sentence)
        ntags = self.grammar.ntags

        print("memory")
        score = np.ones((n, n, ntags), dtype=float)
        back = np.empty((n, n, ntags), dtype="O")

        print("lexical")
        # Apply lexical rules.
        for i, w in enumerate(sentence):
            for t in self.lexicon.tags:
                s = self.lexicon.score(w, t)
                if s is not None:
                    score[i, 0, self.grammar.tag_map[t]] = s
                    back[i, 0, self.grammar.tag_map[t]] = [(i, 0, i, 1)]

        print("decoding")
        _cky.decode(n, ntags, score, back, self.grammar.unaries,
                    self.grammar.binaries)

        # Check to make sure that this is a valid parse.
        root = back[0][-1][self.grammar.tag_map[root_tag]]
        if root is None:
            print(score[0][-1])
            raise RuntimeError("Invalid parse")

        # Build the tree using the backpointers.
        tree = nltk.Tree(root_tag, [self.build_tree(sentence, back, r)
                                    for r in root])
        return tree

    def update_unaries(self, sentence, start, end, score, back):
        dist = score[start][end]
        added = True
        while added:
            added = False
            for child, value in enumerate(dist):
                if value > 0:
                    continue
                for parent in range(self.grammar.ntags):
                    prob = self.grammar.unaries[parent, child]
                    if prob > 0:
                        continue
                    p = value + prob
                    if dist[parent] > 0 or p > dist[parent]:
                        added = True
                        dist[parent] = p
                        back[start][end][parent] = [
                            (start, end, child, False)
                        ]

    def build_tree(self, sentence, back, coords):
        ix, iy, name, is_terminal = coords
        if is_terminal:
            return sentence[name]
        nodes = back[ix][iy][name]
        return nltk.Tree(self.grammar.all_tags[name],
                         [self.build_tree(sentence, back, n) for n in nodes])


class Grammar(object):

    def setup(self, unaries, binaries):
        # Compute the normalizations of the emissions.
        norms = defaultdict(float)
        for k, dist in unaries.items():
            norms[k] += sum(dist.values())
        for k, dist in binaries.items():
            norms[k] += sum(dist.values())

        # Normalize the probabilities.
        [unaries[k].normalize(norms[k]) for k in unaries]
        [binaries[k].normalize(norms[k]) for k in binaries]

        # Determine all the known tags.
        all_tags = [k for k in unaries] + [k for k in binaries]
        all_tags += [k for u in unaries.values() for k in u.keys()]
        all_tags += [k for b in binaries.values() for children in b.keys()
                     for k in children.split()]
        self.all_tags = list(set(all_tags))
        self.tag_map = dict([(k, i) for i, k in enumerate(self.all_tags)])

        # Build the probability matrices.
        ntags = self.ntags = len(self.tag_map)
        print("Using {0} tags".format(ntags))
        self.unaries = 99. + np.zeros((ntags, ntags))
        for parent, u in unaries.items():
            pind = self.tag_map[parent]
            for child, p in u.items():
                self.unaries[self.tag_map[child], pind] = p

        self.binaries = 99. + np.zeros((ntags, ntags, ntags))
        for parent, b in binaries.items():
            pind = self.tag_map[parent]
            for children, p in b.items():
                left_child, right_child = children.split()
                self.binaries[self.tag_map[left_child],
                              self.tag_map[right_child], pind] = p


class MiniGrammar(Grammar):

    def __init__(self, rules):
        unaries = defaultdict(Counter)
        binaries = defaultdict(Counter)

        for l, r, p in rules:
            if len(r.split()) == 1:
                unaries[l][r] += p
            else:
                binaries[l][r] += p

        super(MiniGrammar, self).setup(unaries, binaries)


class FullGrammar(Grammar):

    def __init__(self, corpus, lexicon, max_train=999, horizontal=None,
                 vertical=0):
        unaries = defaultdict(Counter)
        binaries = defaultdict(Counter)

        for i, tree in enumerate(corpus.parsed_sents()):
            if len(tree.leaves()) > max_train:
                continue
            tree.chomsky_normal_form(horzMarkov=horizontal,
                                     vertMarkov=vertical)
            for p in tree.productions():
                node = unicode(p.lhs())
                children = map(unicode, p.rhs())
                if p.is_lexical():
                    lexicon.update(children[0], node)
                    continue
                assert len(children) in [1, 2], "{0} {1}".format(p, i)
                if len(children) == 1:
                    unaries[node][children[0]] += 1
                else:
                    binaries[node]["{0} {1}".format(*children)] += 1

        super(FullGrammar, self).setup(unaries, binaries)


class Lexicon(object):

    def __init__(self, tagged_words):
        self.tag_types = Counter()
        self.tags = Counter()
        self.words = Counter()
        self.words_to_tags = defaultdict(Counter)
        self.total = 0

    def update(self, word, tag):
        # Unknown word model.
        if word in self.words:
            self.tag_types[tag] += 1
        self.tags[tag] += 1
        self.words[word] += 1
        self.words_to_tags[word][tag] += 1
        self.total += 1

    def finalize(self):
        self.tag_types.normalize()
        self.tags.normalize()

    def score(self, word, tag):
        if tag not in self.tags:
            return None

        c_word = self.words[word]
        c_tag_and_word = self.words_to_tags[word][tag]
        if c_word < 10:
            c_word += 1
            c_tag_and_word += exp(self.tag_types[tag])
        p_word = (1 + c_word) / (1 + self.total)
        p_tag_given_word = c_tag_and_word / c_word

        lnp_tag = self.tags[tag]
        try:
            return log(p_tag_given_word) + log(p_word) - lnp_tag
        except ValueError:
            return None

    def best_tag(self, word):
        tags = self.tags.keys()
        scores = [(t, self.score(word, t)) for t in tags]
        return max(scores, key=lambda e: e[1])[0]


class MiniLexicon(Lexicon):

    def __init__(self, rules):
        self.tags = defaultdict(Counter)
        for t, w, p in rules:
            self.tags[t][w] += p
        [self.tags[k].normalize() for k in self.tags]

    def score(self, word, tag):
        t = self.tags[tag]
        return t[word] if word in t else None
