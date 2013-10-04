#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["read_dataset", "TrigramScorer", "StupidUnknownWordModel",
           "UnknownWordModel", "POSTagger"]

import re
import random
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


class TrigramScorer(object):

    def __init__(self, unknown, lambda2=0.3, lambda3=0.6):
        self.unknown = unknown

        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.p_word_tag = defaultdict(lambda: Counter(float))
        self.p_tag = Counter(float)
        self.p_tag_ptag = defaultdict(lambda: Counter(float))
        self.p_tag_pptag = defaultdict(lambda: Counter(float))

        self.trigram_count = 0
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)

    def estimate_lambdas(self):
        l = [0, 0, 0]

        trigrams = self.trigram_counts.keys()
        random.shuffle(trigrams)

        c1, c2, c3 = [], [], []
        for tg in trigrams:
            t1, t2, t3 = tg.split()
            bg1 = self.bigram_counts[" ".join([t1, t2])] - 1
            c1.append(((self.trigram_counts[tg] - 1) / bg1)
                      if bg1 > 0 else 0)

            ug1 = self.unigram_counts[t2] - 1
            c2.append(((self.bigram_counts[" ".join([t2, t3])]-1) / ug1)
                      if ug1 > 0 else 0)

            c3.append((self.unigram_counts[t3]-1)/(self.trigram_count-1))

        for tg, cases in zip(trigrams, zip(c1, c2, c3)):
            count = self.trigram_counts[tg]
            l[max(enumerate(cases), key=lambda o: o[1])[0]] += count

        norm = sum(l)
        self.lambda2 = l[1] / norm
        self.lambda3 = l[0] / norm
        print("Estimated lambda_2 = {0}".format(self.lambda2))
        print("Estimated lambda_3 = {0}".format(self.lambda3))

    def train(self, sentences, threshold=10):
        trigrams = extract_all_trigrams(sentences)
        self.words = set([])
        word_counts = defaultdict(int)
        for trigram in trigrams:
            ppt, pt, t, w = trigram

            self.trigram_count += 1
            self.unigram_counts[t] += 1
            self.bigram_counts[" ".join([pt, t])] += 1
            self.trigram_counts[" ".join([ppt, pt, t])] += 1

            self.words.add(w)
            word_counts[w] += 1

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

        # Build the unknown word model using the uncommon words.
        rare_words = set([w for w, c in word_counts.items() if c <= threshold])
        training_data = [(w, t) for sentence in sentences
                         for w, t in sentence if w in rare_words]
        print("Training unknown word classifier with {0} examples"
              .format(len(training_data)))
        self.unknown.train(training_data)

    def trigram_scores(self, trigram):
        tags = self.p_tag.keys()
        scores = self.trigram_scores_list(trigram, tags)
        return [(t, s) for t, s in zip(tags, scores) if s is not None]

    def trigram_scores_list(self, trigram, tags):
        # Parse the trigram and deal with unknown words.
        ppt, pt, w = trigram
        if w not in self.words:
            word_prob = self.unknown.get_log_probs(tags, w)

        else:
            word_prob = [log(self.p_word_tag[t][w])
                         if w in self.p_word_tag[t] else None
                         for t in tags]

        # Loop over possible tags and compute the scores.
        l2, l3 = self.lambda2, self.lambda3
        p1 = self.p_tag
        p2 = self.p_tag_ptag[pt]
        p3 = self.p_tag_pptag[" ".join([ppt, pt])]
        return [log((1-l2-l3)*p1[t] + l2*p2[t] + l3*p3[t]) + wp
                if wp is not None else None
                for t, wp in zip(tags, word_prob)]


class StupidUnknownWordModel(object):

    def __init__(self):
        self.p0 = Counter(float)

    def train(self, data):
        for w, t in data:
            self.p0[t] += 1
        self.p0.normalize()

    def get_log_probs(self, tags, word):
        return [log(self.p0[t]) if t in self.p0 else None for t in tags]


class UnknownWordModel(object):

    _re = re.compile("[0-9]")

    def __init__(self, maxn=5, lambda1=0.1, lambda2=0.3, lambda3=0.5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.p0 = Counter(float)
        self.p1 = defaultdict(lambda: Counter(float))
        self.p2 = defaultdict(lambda: Counter(float))
        self.p3 = defaultdict(lambda: Counter(float))

        self.pc = defaultdict(float)

        self.maxn = maxn
        self.pn = defaultdict(lambda: [0 for n in range(maxn)])

    def train(self, data):
        # Build up the empirical probabilities.
        for w, t in data:
            self.p0[t] += 1
            self.p1[t][w[-1]] += 1
            self.p2[t][w[-2:]] += 1
            self.p3[t][w[-3:]] += 1
            if w[0].lower() != w[0]:
                self.pc[t] += 1
            n = min([len(self._re.findall(w)), self.maxn-1])
            self.pn[t][n] += 1

        # Normalize the distributions.
        self.p0.normalize()
        [p.normalize() for t, p in self.p1.items()]
        [p.normalize() for t, p in self.p2.items()]
        [p.normalize() for t, p in self.p3.items()]
        for t in self.p0.keys():
            self.pc[t] = max([self.pc[t], 1.0]) / len(data)
        for t, p in self.pn.items():
            norm = sum([v+1 for v in p])
            self.pn[t] = [(v + 1) / norm for v in p]

    def get_log_probs(self, tags, word):
        l1, l2, l3 = self.lambda1, self.lambda2, self.lambda3
        l0 = 1.0 - l1 - l2 - l3
        s1, s2, s3 = word[-1], word[-2:], word[-3:]
        n = min([len(self._re.findall(word)), self.maxn-1])
        cap = word[0].lower() != word[0]
        return [log(l0*self.p0[t] + l1*self.p1[t][s1] + l2*self.p2[t][s2] +
                    l3*self.p3[t][s3])
                + log(self.pn[t][n])
                + log(self.pc[t] if cap else 1-self.pc[t])
                if self.pn[t][n] > 0 else None
                for t in tags]


class POSTagger(object):

    def __init__(self, scorer, greedy=False):
        self.scorer = scorer
        self.tags = [START] + self.scorer.p_tag.keys()
        if greedy:
            self.decode = self.greedy_decode
        else:
            self.decode = self.viterbi_decode

    def greedy_decode(self, sentence):
        states = [self.tags.index(START), self.tags.index(START)]
        for word in sentence + (STOP, STOP):
            scores = max(zip(self.tags,
                             self._score_func(*(states[-2:] + [word]))),
                         key=lambda v: v[1])
            states.append(self.tags.index(scores[0]))
        return [self.tags[i] for i in states[2:-2]]

    def viterbi_decode(self, sentence):
        tags = viterbi(len(self.tags), list(sentence) + [STOP, STOP],
                       self._score_func)
        return [self.tags[i] for i in tags[1:]]

    def _score_func(self, ind1, ind2, word):
        ppt, pt = self.tags[ind1], self.tags[ind2]
        return self.scorer.trigram_scores_list([ppt, pt, word], self.tags)

    def score_tagging(self, words, tags):
        tags = [START, START] + list(tags) + [STOP, STOP]
        ninf = -float("inf")
        return sum([dict(self.scorer.trigram_scores(tags[i:i+2]
                                                    + [word])).get(tags[i+2],
                                                                   ninf)
                    for i, word in enumerate(list(words) + [STOP, STOP])])

    def test(self, sentences, check_sub=False, outfile=None):
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
            if check_sub and outfile is None:
                gold_score = self.score_tagging(words, gold)
                guess_score = self.score_tagging(words, guess)
                if gold_score > guess_score:
                    sub += 1

        if sub > 0:
            print("Suboptimalities detected: {0}".format(sub))

        return correct / total, unk / unk_total
