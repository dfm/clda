#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import os
import hw3
import argparse

parser = argparse.ArgumentParser(
    description="Part of speech tagging.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")
parser.add_argument("-o", "--outfile", default=None,
                    help="The file where the output should be written.")
parser.add_argument("--verbose", action="store_true",
                    help="Print the tag errors.")
parser.add_argument("--greedy", action="store_true",
                    help="Use greedy decoder instead of Viterbi.")
parser.add_argument("--estimate", action="store_true",
                    help="Estimate the interpolation coefficients from "
                    "the training data?")
parser.add_argument("--stupid", action="store_true",
                    help="Use a stupid unknown word model.")
parser.add_argument("--context", action="store_true",
                    help="Use the context-free model.")
parser.add_argument("--lambda2", default=0.3, type=float,
                    help="The bigram weight in the emission model.")
parser.add_argument("--lambda3", default=0.6, type=float,
                    help="The trigram weight in the emission model.")
parser.add_argument("--thresh", default=10, type=float,
                    help="Threshold for maximum number of times a word is "
                    "observed in the training data to count as unknown.")
parser.add_argument("--theta1", default=0.1, type=float,
                    help="The unigram weight in the suffix model.")
parser.add_argument("--theta2", default=0.3, type=float,
                    help="The bigram weight in the suffix model.")
parser.add_argument("--theta3", default=0.5, type=float,
                    help="The trigram weight in the suffix model.")


data_path = "data"

# Load all the datasets.
print("Loading training data")
training_data = hw3.read_dataset(os.path.join(data_path,
                                              "en-wsj-train.pos"))

print("Loading in-domain validation data")
dev_in_data = hw3.read_dataset(os.path.join(data_path, "en-wsj-dev.pos"))

print("Loading out-of-domain validation data")
dev_out_data = hw3.read_dataset(os.path.join(data_path,
                                             "en-web-weblogs-dev.pos"))

print("Loading out-of-domain test data")
test_data = hw3.read_dataset(os.path.join(data_path, "en-web-test.blind"))


def main(in_args):
    args = parser.parse_args(in_args)

    # Choose the unknown word model.
    if args.stupid:
        unk = hw3.StupidUnknownWordModel()
    else:
        unk = hw3.UnknownWordModel(theta1=args.theta1,
                                   theta2=args.theta2,
                                   theta3=args.theta3)

    # Set up and train the local tag scorer.
    print("Training scorer")
    if args.context:
        scorer = hw3.ContextFreeScorer(unk)
    else:
        scorer = hw3.TrigramScorer(unk, lambda2=args.lambda2,
                                   lambda3=args.lambda3)
    nunk = scorer.train(training_data, threshold=args.thresh)
    if args.estimate:
        print("Estimating interpolation coefficients")
        scorer.estimate_lambdas()

    model = hw3.POSTagger(scorer, greedy=args.greedy)

    print("Testing in-domain performance")
    acc_in, unk_in = model.test(dev_in_data, verbose=args.verbose)
    print("Accuracy: {0:.4f}, Unknown: {1:.4f}".format(acc_in*100, unk_in*100))

    print("Testing out-of-domain performance")
    acc_out, unk_out = model.test(dev_out_data, verbose=args.verbose)
    print("Accuracy: {0:.4f}, Unknown: {1:.4f}".format(acc_out*100,
                                                       unk_out*100))

    if args.outfile is not None:
        print("Writing out-of-domain predictions")
        model.test(test_data, outfile=args.outfile)

    return nunk, acc_in, unk_in, acc_out, unk_out


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
