#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

from collections import defaultdict
from itertools import product, izip, izip_longest

ALIGNMENT_TYPES = [None, "P", "S"]


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
        [f.write(" ".join(["{0}-{1}".format(*p)
                           for p in proposal if p[0] >= 0]) + " \n")
         for proposal in proposals]


class BaselineWordAligner(object):

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

    def train(self, pairs):
        self.count_both = defaultdict(lambda: defaultdict(int))
        self.count_en = defaultdict(int)
        self.count_fr = defaultdict(int)
        for pair in pairs:
            for e, f in izip(pair[0], pair[1]):
                self.count_both[e][f] += 1
                self.count_en[e] += 1
                self.count_fr[f] += 1
        print(self.count_en)

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


if __name__ == "__main__":
    # Load the data.
    validation_pairs = read_sentence_pairs("data/trial/trial", alignments=True)
    test_pairs = read_sentence_pairs("data/test/test")

    # Set-up the aligner.
    aligner = HeuristicWordAligner()
    aligner.train(validation_pairs)
    # aligner = BaselineWordAligner()

    # Render the alignments.
    map(aligner.render, validation_pairs)

    # Compute the test statistics on the validation set.
    test(aligner, validation_pairs)

    # Write the predictions.
    predict(aligner, test_pairs, "output.txt")
