#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import os
import hw3


if __name__ == "__main__":
    data_path = "data"

    # Load all the datasets.
    print("Loading training data")
    training_data = hw3.read_dataset(os.path.join(data_path,
                                                  "en-wsj-train.pos"))
    training_vocab = hw3.extract_vocabulary(training_data)

    print("Loading in-domain validation data")
    dev_in_data = hw3.read_dataset(os.path.join(data_path, "en-wsj-dev.pos"))

    print("Loading out-of-domain validation data")
    dev_out_data = hw3.read_dataset(os.path.join(data_path,
                                                 "en-web-weblogs-dev.pos"))

    print("Loading out-of-domain test data")
    test_data = hw3.read_dataset(os.path.join(data_path, "en-web-test.blind"))

    # Set up and train the local tag scorer.
    print("Training scorer")
    scorer = hw3.FrequentTagScorer()
    scorer.train(hw3.extract_all_trigrams(training_data))
    model = hw3.POSTagger(training_vocab, scorer)

    print("Testing in-domain performance")
    acc, unk = model.test(dev_in_data)
    print("Accuracy: {0:.4f}, Unknown: {1:.4f}".format(acc*100, unk*100))

    print("Testing out-of-domain performance")
    acc, unk = model.test(dev_out_data)
    print("Accuracy: {0:.4f}, Unknown: {1:.4f}".format(acc*100, unk*100))

    print("Writing out-of-domain predictions")
    model.test(test_data, outfile="output.txt")
