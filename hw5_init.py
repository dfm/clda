#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import hw5
import argparse
import cPickle as pickle
from nltk.corpus.reader import BracketParseCorpusReader

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data")
parser.add_argument("--maxTrain", default=999, type=int)
parser.add_argument("--horizontal", default=None, type=int)
parser.add_argument("--vertical", default=0, type=int)
parser.add_argument("--mini", action="store_true")
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
    # Load the training data.
    print("Loading training trees")
    train_corpus = BracketParseCorpusReader(args.data,
                                            "en-wsj-train.1.mrg")

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
                                      root_tag="S")
    tree.draw()

else:
    print("Parsing")
    s = train_corpus.sents()[3]
    print(len(s))
    tree = parser.generate_parse_tree(s)
    tree.un_chomsky_normal_form()
    tree.draw()
