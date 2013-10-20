#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import random
import numpy as np
from glob import glob
from collections import defaultdict
from itertools import izip, izip_longest, product

NULL_TOKEN = "<NULL>"
ALIGNMENT_TYPES = [None, "P", "S"]


def default_index(x, a, d):
    try:
        return x.index(a)
    except ValueError:
        return d


def index_or_append(obj, el):
    try:
        return obj.index(el)
    except ValueError:
        obj.append(el)
        return len(obj) - 1


def parse_sentence(line):
    _id = -1
    words = []
    for word in line.split():
        w = word.strip()
        if w == "<s" or w == "</s>":
            continue
        elif w.startswith("snum="):
            _id = int(w[5:-1])
            continue
        words.append(w)
    return _id, words


def read_sentence_pairs(base_path, alignments=False):
    with open("{0}.e".format(base_path)) as f:
        en_sentences = [parse_sentence(line.decode("latin-1").strip())
                        for line in f]
    with open("{0}.f".format(base_path)) as f:
        fr_sentences = [parse_sentence(line.decode("latin-1").strip())
                        for line in f]
    if alignments:
        with open("{0}.wa".format(base_path)) as f:
            alignment_list = [line.strip().split() for line in f]

    # Zip the sentence pairs.
    ids = []
    pairs = []
    for e, f in zip(en_sentences, fr_sentences):
        assert e[0] == f[0], "id mismatch"
        align = None
        if alignments:
            align = [[0 for i in range(len(f[1]))] for j in range(len(e[1]))]
        ids.append(e[0])
        pairs.append([e[1], f[1], align])

    # Build the alignment matrix.
    if alignments:
        for a in alignment_list:
            _id, ep, fp = map(int, a[:-1])
            t = a[-1]
            pairs[ids.index(_id)][2][ep-1][fp-1] = ALIGNMENT_TYPES.index(t)

    return pairs


def load_training_pairs(basepath, maxn):
    fns = glob(os.path.join(basepath, "*.e"))
    if maxn is not None:
        random.shuffle(fns)
    pairs = []
    for fn in fns:
        print("  ... File: {0}".format(fn))
        pairs += read_sentence_pairs(os.path.splitext(fn)[0])
        if maxn is not None and len(pairs) > maxn:
            break
    if maxn is None:
        return pairs
    return pairs[:maxn]


def test(aligner, pairs):
    prop_poss_count = 0
    prop_sure_count = 0
    prop_count = 0
    sure_count = 0
    proposals = map(aligner.align, pairs)
    for pair, proposal in zip(pairs, proposals):
        ref = pair[2]
        sure_count += sum([1 for row in ref for v in row if v == 2])
        for p in proposal:
            if p[0] >= 0:
                prop_count += 1
                val = ref[p[0]][p[1]]
                # print(val)
                if val == 1:
                    prop_poss_count += 1
                elif val == 2:
                    prop_poss_count += 1
                    prop_sure_count += 1
    print("Precision: {0}".format(prop_poss_count / prop_count))
    print("Recall: {0}".format(prop_sure_count / sure_count))
    print("AER: {0}".format(1.0 - (prop_sure_count + prop_poss_count)
                            / (sure_count + prop_count)))


def predict(aligner, pairs, fn):
    proposals = map(aligner.align, pairs)
    with open(fn, "w") as f:
        [f.write(" ".join(["{1}-{0}".format(*p)
                           for p in proposal if p[0] >= 0]) + " \n")
         for proposal in proposals]


class BaselineWordAligner(object):

    def train(self, pairs, **kwargs):
        pass

    def align(self, pair):
        en, fr, al = pair
        el, fl = len(en), len(fr)
        return [(fi if fi < el else -1, fi) for fi in range(fl)]

    def render(self, pair):
        alignment = self.align(pair)
        en, fr, ref = pair
        for fi, (f, al) in enumerate(zip(fr, alignment)):
            for ei in range(len(en)):
                fmt = " {0} "
                c = " "
                if ei == al[0]:
                    c = "#"
                if ref is not None:
                    if ref[ei][fi] == 1:
                        fmt = "({0})"
                    elif ref[ei][fi] == 2:
                        fmt = "[{0}]"
                print(fmt.format(c), end="")
            print("| {0}".format(f))
        print("---" * len(en) + "'")
        for ec in izip_longest(*en, fillvalue=" "):
            print("".join(map(" {0} ".format, ec)))
        print()


class HeuristicWordAligner(BaselineWordAligner):

    def train(self, pairs, **kwargs):
        self.count_both = defaultdict(lambda: defaultdict(int))
        self.count_en = defaultdict(int)
        self.count_fr = defaultdict(int)
        for pair in pairs:
            for e, f in izip(pair[0], pair[1]):
                self.count_both[e][f] += 1
                self.count_en[e] += 1
                self.count_fr[f] += 1

    def align(self, pair):
        cb = self.count_both
        ce = self.count_en
        cf = self.count_fr
        result = []
        for fi, f in enumerate(pair[1]):
            probs = [(ei, cb[e][f]/(ce[e]*cf[f]))
                     for ei, e in enumerate(pair[0])
                     if e in ce and f in cf]
            if not len(probs):
                result.append((-1, fi))
                continue
            result.append((max(probs, key=lambda o: o[1])[0], fi))

        return result


class IBMModel1Aligner(BaselineWordAligner):

    def __init__(self, nullprob=0.2):
        self.nullprob = nullprob

    def train(self, pairs, niter=40):
        # Figure out the vocabularies.
        self.vocab_en = defaultdict(lambda: len(self.vocab_en))
        self.vocab_fr = defaultdict(lambda: len(self.vocab_fr))
        index_pairs = []
        for pair in pairs:
            index_pairs.append((
                np.array([self.vocab_en[w.lower()]
                          for w in pair[0]] + [-1]),
                np.array([self.vocab_fr[w.lower()]
                          for w in pair[1]])
            ))

        print(self.vocab_en[NULL_TOKEN])
        self.vocab_en = dict(self.vocab_en)
        self.vocab_fr = dict(self.vocab_fr)

        self.run_em(index_pairs, niter)

    def run_em(self, pairs, niter):
        # Initialize the p(f | e) as a uniform distribution.
        self.prob_fe = np.ones((len(self.vocab_en), len(self.vocab_fr)))
        self.prob_fe /= np.sum(self.prob_fe, axis=1)[:, None]
        for i in range(niter):
            print("EM iteration {0}...".format(i))
            counts = np.zeros_like(self.prob_fe)
            for pair in pairs:
                inds = (pair[0][:, None], pair[1][None, :])
                counts[inds] += (self.prob_fe[inds]
                                 / np.sum(self.prob_fe[inds], axis=1)[:, None])

            # Normalize the counts to get the ML pair probabilities.
            self.prob_fe = counts / np.sum(counts, axis=1)[:, None]
            print(np.sum(self.prob_fe))

    def _alignment_prob(self, pair):
        # Compute the alignment probability (uniform).
        align_prob = (1-self.nullprob)*np.ones(len(pair[0]))/(len(pair[0])-1)

        # Add extra probability for the null alignment.
        align_prob[-1] = self.nullprob

        return align_prob[:, None]

    def _get_rel_post(self, pair):
        align_prob = self._alignment_prob(pair)

        # Compute the (relative) posterior probabilities for all
        # possible alignments.
        inds = (pair[0][:, None], pair[1][None, :])
        return self.prob_fe[inds] * align_prob, inds

    def align(self, pair):
        en_inds = np.array([self.vocab_en.get(w.lower(), -1)
                            for w in pair[0]] + [-1])
        fr_inds = np.array([self.vocab_fr.get(w.lower(), -1)
                            for w in pair[1]])
        apost, inds = self._get_rel_post((en_inds, fr_inds))
        ei = np.argmax(apost, axis=0)

        # Deal with NULL alignments.
        ei[ei == len(pair[0])] = -1

        return zip(ei, np.arange(len(pair[1])))


class IBMModel2Aligner(IBMModel1Aligner):

    def __init__(self, alpha, *args, **kwargs):
        super(IBMModel2Aligner, self).__init__(*args, **kwargs)
        self.alpha = alpha

    def run_em(self, pairs, niter):
        # Initialize the p(f | e) as a uniform distribution.
        self.prob_fe = np.ones((len(self.vocab_en), len(self.vocab_fr)))
        self.prob_fe /= np.sum(self.prob_fe, axis=1)[:, None]
        for i in range(niter):
            print("EM iteration {0}...".format(i))
            counts = np.zeros_like(self.prob_fe)
            num, denom = 0.0, 0.0
            for pair in pairs:
                align_prob, d = self._alignment_prob(pair, True)
                inds = (pair[0][:, None], pair[1][None, :])
                p = align_prob * self.prob_fe[inds]
                p /= np.sum(p, axis=1)[:, None]
                counts[inds] += p
                num += np.sum(p[:-1].flatten())
                denom += np.sum(d[:-1].flatten() * p[:-1].flatten())

            # Update alpha.
            self.alpha = num / denom
            print(self.alpha)

            # Normalize the counts to get the ML pair probabilities.
            self.prob_fe = counts / np.sum(counts, axis=1)[:, None]
            print(np.sum(self.prob_fe))

    def _alignment_prob(self, pair, get_deltas=False):
        d = np.abs(np.arange(0.0, float(len(pair[0])), 1.0)[:, None]
                   - np.arange(0.0, float(len(pair[1])), 1.0)[None, :])
        d *= len(pair[0])/len(pair[1])
        align_prob = np.exp(-self.alpha*d)
        nullprob = self.nullprob
        align_prob *= (1-nullprob)/np.sum(align_prob[:-1], axis=0)[None, :]
        align_prob[-1, :] = nullprob
        if get_deltas:
            return align_prob, d
        return align_prob


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Part of speech tagging.")
    parser.add_argument("-d", "--data", default="data",
                        help="The base path for the data files.")
    parser.add_argument("--model", default="baseline",
                        help="The alignment model to use")
    parser.add_argument("--test", action="store_true",
                        help="Run the test experiment")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Render the validation alignments")
    parser.add_argument("-n", "--number", type=int, default=1000,
                        help="The number of training sentences")
    parser.add_argument("-i", "--niter", type=int, default=40,
                        help="The number of EM iterations to run")
    args = parser.parse_args()

    if args.test:
        validation_pairs = read_sentence_pairs("{0}/mini/mini"
                                               .format(args.data),
                                               alignments=True)
        training_pairs = validation_pairs

    else:
        # Load the data.
        print("Loading {0} training sentence pairs.".format(args.number))
        training_pairs = load_training_pairs("{0}/training".format(args.data),
                                             args.number)
        validation_pairs = read_sentence_pairs("{0}/trial/trial"
                                               .format(args.data),
                                               alignments=True)
        test_pairs = read_sentence_pairs("data/test/test")

    # Set-up the aligner.
    aligner = BaselineWordAligner()
    if args.model.lower() == "heuristic":
        aligner = HeuristicWordAligner()
    elif args.model.lower() == "model1":
        aligner = IBMModel1Aligner()
    elif args.model.lower() == "model2":
        aligner = IBMModel2Aligner(0.1)

    print("Training word alignment model.")
    aligner.train(training_pairs, niter=args.niter)

    # Render the alignments.
    if args.verbose:
        map(aligner.render, validation_pairs)

    # Compute the test statistics on the validation set.
    test(aligner, validation_pairs)

    # Write the predictions.
    if not args.test:
        predict(aligner, test_pairs, "output.txt")
