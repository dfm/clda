#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import hw5
import argparse
from nltk.corpus.reader import BracketParseCorpusReader

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data")
parser.add_argument("--maxTrain", default=999, type=int)
parser.add_argument("--maxValid", default=40, type=int)
parser.add_argument("--horizontal", default=None, type=int)
parser.add_argument("--vertical", default=0, type=int)
parser.add_argument("--mini", action="store_true")
parser.add_argument("--theta", default=0.5, type=float)
parser.add_argument("--test", action="store_true")
parser.add_argument("--validate", action="store_true")
args = parser.parse_args()

if args.mini:
    grammar = hw5.MiniGrammar([
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
    lexicon = hw5.MiniLexicon([
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
    print("Loading training trees")
    train_corpus = BracketParseCorpusReader(args.data,
                                            "en-wsj-train.1.mrg")

    if args.validate:
        print("Loading in-domain validation trees")
        in_domain = BracketParseCorpusReader(args.data,
                                             "en-wsj-dev.2.mrg")

        print("Loading out-of-domain validation trees")
        out_of_domain = BracketParseCorpusReader(args.data,
                                                 "en-web-dev.3.mrg")

    if args.test:
        print("Loading test sentences")
        test_sentences = [line.strip().split()
                          for line in open(os.path.join(args.data,
                                                        "en-web-weblogs-test"
                                                        ".sentences"))]

    print("Building grammar and lexicon")
    lexicon = hw5.Lexicon(train_corpus.tagged_words())
    grammar = hw5.FullGrammar(train_corpus, lexicon, max_train=args.maxTrain,
                              horizontal=args.horizontal,
                              vertical=args.vertical)
    lexicon.finalize()

# Set up and train the parser.
parser = hw5.Parser(grammar, lexicon)
# pickle.dump(grammar, open("grammar.pkl", "wb"), -1)

if args.mini:
    tree = parser.generate_parse_tree(["fish", "people", "fish", "tanks"],
                                      root_tag="S", theta=args.theta)
    tree.draw()

else:
    if args.validate:
        validation_fn = os.path.join(args.data, "validation.txt")
        open(validation_fn, "a").write(("h: {0} v: {1} theta: {2} "
                                        "maxTrain: {3} maxValid: {4}\n")
                                       .format(args.horizontal,
                                               args.vertical,
                                               args.theta,
                                               args.maxTrain, args.maxValid))

        evaluator = hw5.Evaluator(["ROOT", "TOP"], ["''", "``", ".", ":", ","])
        for gold in out_of_domain.parsed_sents():
            sentence = map(unicode, gold.leaves())
            if len(sentence) > args.maxValid:
                continue
            guess = parser.generate_parse_tree(sentence, theta=args.theta)
            guess.un_chomsky_normal_form()
            evaluator(guess, gold)
            print("F1 = {0}".format(evaluator.get_f1()))
        open(validation_fn, "a").write("out-of-domain: {0}\n"
                                       .format(evaluator.get_f1()))

        evaluator = hw5.Evaluator(["ROOT", "TOP"], ["''", "``", ".", ":", ","])
        for gold in in_domain.parsed_sents():
            sentence = map(unicode, gold.leaves())
            if len(sentence) > args.maxValid:
                continue
            guess = parser.generate_parse_tree(sentence, theta=args.theta)
            guess.un_chomsky_normal_form()
            evaluator(guess, gold)
            print("F1 = {0}".format(evaluator.get_f1()))
        open(validation_fn, "a").write("in-domain: {0}\n"
                                       .format(evaluator.get_f1()))

    if args.test:
        print("Parsing test set.")
        outfn = os.path.join(args.data, "output.txt")
        open(outfn, "w").close()
        for i, s in enumerate(test_sentences):
            print("Test sentence {0} ({1} words)".format(i, len(s)))
            tree = parser.generate_parse_tree(s, theta=args.theta)
            tree.un_chomsky_normal_form()
            open(outfn, "a").write(tree.pprint(margin=1e10) + "\n")
