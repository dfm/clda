#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
from hw3.scorer import MostFrequentTagScorer
from hw3.data import read_tagged_sentences, get_vocabulary


if __name__ == "__main__":
    data_path = "data"

    # Load all the datasets.
    print("Loading training data")
    training_data = read_tagged_sentences(os.path.join(data_path,
                                                       "en-wsj-train.pos"))
    training_vocab = get_vocabulary(training_data)

    print("Loading in-domain validation data")
    dev_in_data = read_tagged_sentences(os.path.join(data_path,
                                                     "en-wsj-dev.pos"))

    print("Loading out-of-domain validation data")
    dev_out_data = read_tagged_sentences(
        os.path.join(data_path, "en-web-weblogs-dev.pos"))

    print("Loading out-of-domain test data")
    test_data = read_tagged_sentences(os.path.join(data_path,
                                                   "en-web-test.blind"))

    # Set up the local tag scorer.
    local_scorer = MostFrequentTagScorer()
