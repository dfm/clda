#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["read_dataset", "TagScorer", "TrigramScorer", "POSTagger"]

from math import log
from collections import defaultdict

from ._viterbi import viterbi

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

    def train(self, sentences):
        trigrams = extract_all_trigrams(sentences)
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
        return [(t, log(v)) for t, v in probs.items()]


class TrigramScorer(object):

    def __init__(self, lambda2=0.3, lambda3=0.6):
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.p_word_tag = defaultdict(lambda: Counter(float))
        self.p_tag = Counter(float)
        self.p_tag_ptag = defaultdict(lambda: Counter(float))
        self.p_tag_pptag = defaultdict(lambda: Counter(float))

    def train(self, sentences):
        trigrams = extract_all_trigrams(sentences)
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
        p2 = self.p_tag_ptag[pt]
        p3 = self.p_tag_pptag[" ".join([ppt, pt])]
        return [(t, log((1-l2-l3)*tag_prior + l2*p2[t] + l3*p3[t])
                 + log(self.p_word_tag[t][w]))
                for t, tag_prior in self.p_tag.items()
                if w in self.p_word_tag[t]]

    def trigram_scores_list(self, trigram, tags):
        # Parse the trigram and deal with unknown words.
        ppt, pt, w = trigram
        if w not in self.words:
            w = UNKNOWN

        # Loop over possible tags and compute the scores.
        l2, l3 = self.lambda2, self.lambda3
        p1 = self.p_tag
        p2 = self.p_tag_ptag[pt]
        p3 = self.p_tag_pptag[" ".join([ppt, pt])]
        return [log((1-l2-l3)*p1[t] + l2*p2[t] + l3*p3[t])
                + log(self.p_word_tag[t][w])
                if w in self.p_word_tag[t] else None
                for t in tags]


class State(object):

    def __init__(self, ppt, pt):
        self.ppt = ppt
        self.pt = pt


def _state_id(t1, t2):
    return t1 + " " + t2


class POSTagger(object):

    def __init__(self, scorer):
        self.scorer = scorer
        self.tags = [START] + self.scorer.p_tag.keys()

    def greedy_decode(self, sentence):
        states = [START, START]
        for word in sentence + (STOP, STOP):
            scores = max(self.scorer.trigram_scores(states[-2:] + [word]),
                         key=lambda v: v[1])
            states.append(scores[0])
        return states[2:-2]

    def _score_func(self, ind1, ind2, word):
        ppt, pt = self.tags[ind1], self.tags[ind2]
        return self.scorer.trigram_scores_list([ppt, pt, word], self.tags)

    def decode(self, sentence):
        tags = viterbi(len(self.tags), list(sentence) + [STOP, STOP],
                       self._score_func)
        return [self.tags[i] for i in tags[1:]]

    def score_tagging(self, words, tags):
        tags = [START, START] + list(tags) + [STOP, STOP]
        ninf = -float("inf")
        return sum([dict(self.scorer.trigram_scores(tags[i:i+2]
                                                    + [word])).get(tags[i+2],
                                                                   ninf)
                    for i, word in enumerate(list(words) + [STOP, STOP])])

    def test(self, sentences, outfile=None):
        if outfile is not None:
            open(outfile, "w")

        vocab = self.scorer.words
        correct, total = 0, 0
        unk, unk_total = 0, 0
        sub = 0
        for sentence in sentences:
            words, gold = zip(*sentence)
            guess = self.decode(words)

            if outfile is not None:
                with open(outfile, "a") as f:
                    [f.write("{0}\t{1}\n".format(w, g))
                     for w, g in zip(words, guess)]
                    f.write("\n")

            correct += sum([t1 == t2 for t1, t2 in zip(gold, guess)])
            total += len(words)

            # Check unknown word accuracy.
            tmp = zip(*[[gl == gu, 1] for w, gl, gu in zip(words, gold, guess)
                        if w not in vocab])
            if len(tmp):
                unk += sum(tmp[0])
                unk_total += sum(tmp[1])

            # Check for decoding suboptimalities.
            if outfile is None:
                gold_score = self.score_tagging(words, gold)
                guess_score = self.score_tagging(words, guess)
                if gold_score > guess_score:
                    sub += 1
                    print(gold)
                    print(guess)
                    print(words)
                    print(gold_score, guess_score)

        if sub > 0:
            print("Suboptimalities detected: {0}".format(sub))

        return correct / total, unk / unk_total
