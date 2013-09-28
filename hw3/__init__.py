#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
from collections import defaultdict

START = "<S>"
STOP = "</S>"
UNKNOWN = "<UNK>"


class Counter(defaultdict):

    def normalize(self):
        norm = sum(self.values())
        [self.__setitem__(k, v / norm) for k, v in self.items()]


def read_dataset(path):
    sentences = [[]]
    with open(path) as f:
        for line in f:
            if line == "\n":
                sentences.append([])
                continue
            columns = line.strip().split("\t")
            word, tag = columns if len(columns) > 1 else (columns[0], None)
            sentences[-1].append((word, tag))
    return [s for s in sentences if len(s)]


def extract_vocabulary(sentences):
    vocab = set([])
    [vocab.update(set([w for w, t in s])) for s in sentences]
    return vocab


def extract_trigrams(sentence):
    sentence = ([(START, START), (START, START)]
                + sentence
                + [(STOP, STOP), (STOP, STOP)])
    return [(sentence[i][1], sentence[i+1][1], t[1], t[0])
            for i, t in enumerate(sentence[2:])]


def extract_all_trigrams(sentences):
    return [t for s in map(extract_trigrams, sentences) for t in s]


class TagScorer(object):

    def __init__(self):
        self.words = defaultdict(lambda: Counter(float))
        self.unknown = Counter(float)

    def train(self, trigrams):
        self.seen = []
        for trigram in trigrams:
            ppt, pt, t, w = trigram
            if w not in self.words:
                self.unknown[t] += 1
            self.words[w][t] += 1
            self.seen.append(" ".join([ppt, pt, t]))

        self.seen = set(self.seen)
        [w.normalize() for k, w in self.words.items()]
        self.unknown.normalize()

    def trigram_scores(self, trigram):
        ppt, pt, w = trigram
        probs = self.words[w] if w in self.words else self.unknown
        # allowed = [t for t in probs if " ".join([ppt, pt, t]) in self.seen]
        return [(t, np.log(v)) for t, v in probs.items()]


class TrigramScorer(object):

    def __init__(self):
        self.p_word_tag = defaultdict(lambda: Counter(float))
        self.p_tag = Counter(float)
        self.p_tag_ptag = defaultdict(lambda: Counter(float))
        self.p_tag_pptag = defaultdict(lambda: Counter(float))

    def train(self, trigrams):
        self.words = set([])
        for trigram in trigrams:
            ppt, pt, t, w = trigram

            # Unknown word model.
            if w not in self.words:
                self.p_word_tag[t][UNKNOWN] += 1

            self.words.add(w)

            # Keep track of the tag prior.
            self.p_tag[t] += 1

            # Update the conditional probabilities.
            self.p_word_tag[t][w] += 1
            self.p_tag_ptag[pt][t] += 1
            self.p_tag_pptag[" ".join([ppt, pt])][t] += 1

        # Normalize the distributions.
        self.p_tag.normalize()
        [dist.normalize() for k, dist in self.p_word_tag.items()]
        [dist.normalize() for k, dist in self.p_tag_ptag.items()]
        [dist.normalize() for k, dist in self.p_tag_pptag.items()]

    def trigram_scores(self, trigram):
        # Parse the trigram and deal with unknown words.
        ppt, pt, w = trigram
        if w not in self.words:
            w = UNKNOWN

        # Loop over possible tags and compute the scores.
        l2, l3 = self.lambda2, self.lambda3
        return [(t, np.log((1-l2-l3)*tag_prior + l2*self.p_tag_ptag[pt][t]
                           + l3*self.p_tag_pptag[" ".join([ppt, pt])][t])
                 + np.log(self.p_word_tag[t][w]))
                for t, tag_prior in self.p_tag.items()]


class POSTagger(object):

    def __init__(self, vocab, scorer):
        self.vocab = set(vocab)
        self.scorer = scorer

    def decode(self, sentence):
        states = [START, START]
        for word in sentence + (STOP, STOP):
            scores = sorted(self.scorer.trigram_scores(states[-2:] + [word]),
                            key=lambda v: v[1])
            states.append(scores[-1][0])
        return states[2:-2]

    def test(self, sentences, outfile=None):
        if outfile is not None:
            open(outfile, "w")

        correct, total = 0, 0
        unk = np.array([0, 0])
        for sentence in sentences:
            words, gold = zip(*sentence)
            guess = self.decode(words)

            if outfile is not None:
                with open(outfile, "a") as f:
                    [f.write("{0}\t{1}\n".format(w, g))
                     for w, g in zip(words, guess)]
                    f.write("\n")

            correct += np.sum([t1 == t2 for t1, t2 in zip(gold, guess)])
            total += len(words)
            tmp = [[gl == gu, 1] for w, gl, gu in zip(words, gold, guess)
                   if w not in self.vocab]
            if len(tmp):
                unk += np.sum(np.array(tmp), axis=0)
        return correct / total, unk[0] / unk[1]
