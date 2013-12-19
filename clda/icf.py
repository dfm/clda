#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["ICF"]

from collections import defaultdict

import numpy as np

from .utils import _function_wrapper


class ICF(object):

    def __init__(self, K, nusers, nitems, alpha=10.0, l2v=0.1, l2u=0.1,
                 theta=None):
        self.K = K
        self.nusers = nusers
        self.nitems = nitems
        self.alpha = alpha
        self.l2v = l2v
        self.l2u = l2u

        self.U = np.random.rand(nusers, K)
        if theta is None:
            self.theta = np.zeros((nitems, K))
            self.V = np.random.rand(nitems, K)
        else:
            self.theta = theta
            self.V = np.array(theta)

    def train(self, training_set, test_set=None, pool=None, lda=False,
              N=range(50, 501, 50)):
        M = map if pool is None else pool.map

        user_items = [[] for i in range(self.nusers)]
        item_users = [[] for i in range(self.nitems)]
        [(user_items[u].append(a), item_users[a].append(u))
         for u, a in training_set]

        if test_set is not None:
            test_user_items = defaultdict(list)
            [test_user_items[u].append(a) for u, a in test_set]
            test_args = [(u, t, user_items[u])
                         for u, t in test_user_items.items()]

        count = 0
        while True:
            print("Updating users")
            vtv = np.dot(self.V.T, self.V)
            self.U = np.vstack(M(_function_wrapper(self,
                                                   "compute_user_update",
                                                   vtv), user_items))

            if lda and count == 0:
                print("Computing LDA recall")
                self.lda_recall = np.mean(M(_function_wrapper(self,
                                                              "compute_recall",
                                                              N=N),
                                            test_args), axis=0)
                print("LDA recall: {0}".format(self.lda_recall))
            count += 1

            print("Updating items")
            utu = np.dot(self.U.T, self.U)
            self.V = np.vstack(M(_function_wrapper(self,
                                                   "compute_item_update",
                                                   utu),
                                 zip(item_users, self.theta)))

            # Compute the held out recall.
            if test_set is not None:
                print("Computing held out recall")
                yield np.mean(M(_function_wrapper(self, "compute_recall", N=N),
                                test_args), axis=0)
            else:
                yield 0.0

    def compute_user_update(self, vec, vtv):
        vm = self.V[vec]
        vtcv = vtv + self.alpha * np.dot(vm.T, vm)
        vtcv[np.diag_indices(self.K)] += self.l2u
        b = (1+self.alpha)*np.sum(vm, axis=0)
        return np.linalg.solve(vtcv, b)

    def compute_item_update(self, args, utu):
        vec, theta = args
        um = self.U[vec]
        utcu = utu + self.alpha * np.dot(um.T, um)
        utcu[np.diag_indices(self.K)] += self.l2v
        b = (1+self.alpha)*np.sum(um, axis=0) + self.l2v*theta
        return np.linalg.solve(utcu, b)

    def compute_recall(self, args, N=range(50, 501, 50)):
        u, items, previous = args

        # Remove items in training data.
        m = np.ones(self.nitems, dtype=bool)
        m[previous] = False

        # Compute the top recommendations.
        r = np.dot(self.V[m], self.U[u])
        inds = np.arange(self.nitems)[m][np.argsort(r)[::-1]]

        # Compute the recall.
        recall = np.cumsum([i in items for i in inds]) / len(items)
        return recall[N]

    def mean_recall(self, training_set, test_set, N=range(50, 501, 50),
                    pool=None):
        M = map if pool is None else pool.map

        # Parse the datasets.
        user_items = [[] for i in range(self.nusers)]
        [user_items[u].append(a) for u, a in training_set]
        test_user_items = defaultdict(list)
        [test_user_items[u].append(a) for u, a in test_set]
        test_args = [(u, t, user_items[u])
                     for u, t in test_user_items.items()]

        return np.mean(M(_function_wrapper(self, "compute_recall", N=N),
                         test_args), axis=0)
