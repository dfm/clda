#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

from . import _cf


class CF(object):

    def __init__(self, ntopics, l2u=0.01, l2v=0.01, alpha=40.0):
        self.ntopics = ntopics
        self.l2u = l2u
        self.l2v = l2v
        self.alpha = alpha

        # Save the initial events.
        self.users = {}
        self.user_items = []
        self.items = {}
        self.item_users = []

    def add_event(self, user, item):
        # Update the mappings.
        if user not in self.users:
            self.users[user] = len(self.users)
            self.user_items.append([])

        if item not in self.items:
            self.items[item] = len(self.items)
            self.item_users.append([])

        # Append the event to the data stream.
        self.user_items[self.users[user]].append(self.items[item])
        self.item_users[self.items[item]].append(self.users[user])

        # Update the counts.
        self.nusers = len(self.users)
        self.nitems = len(self.items)

    def learn(self, events, pool=None):
        [self.add_event(*evt) for evt in events]

        _cf.update(self.U, self.V, self.user_items, self.item_users)

        # Initialize the matrices.
        # self.V = np.random.rand(self.nitems, self.ntopics)
        # self.U = np.random.rand(self.nusers, self.ntopics)

        # Update the users.
        # print("Updating users")
        # self.VTV = np.dot(self.V.T, self.V)
        # map(self.update_user, range(self.nusers))

        # # Update the items.
        # print("Updating items")
        # self.UTU = np.dot(self.U.T, self.U)
        # map(self.update_item, range(self.nitems))

        print(self.recall(pool=pool))
        assert 0

    def recall(self, pool=None, M=100):
        mp = map if pool is None else pool.map
        return np.mean(mp(_function_wrapper(self, "user_recall", M=M),
                          range(self.nusers)))

    def user_recall(self, user_index, M=100):
        u = self.U[user_index]
        r = self.user_items[user_index]
        recs = np.argsort(np.dot(self.V, u))[-M:]
        recall = np.sum([rec in r for rec in recs]) / len(r)
        return recall

    def update_user(self, user_index):
        r = self.user_items[user_index]
        Vm = self.V[r]
        VmTVm = np.dot(Vm.T, Vm)
        m = self.VTV + self.alpha*VmTVm
        m[zip(range(self.ntopics), range(self.ntopics))] += self.l2u
        self.U[user_index] = np.linalg.solve(m, self.alpha*np.sum(Vm, axis=0))

    def update_item(self, item_index):
        r = self.item_users[item_index]
        Um = self.U[r]
        UmTUm = np.dot(Um.T, Um)
        m = self.UTU + self.alpha*UmTUm
        m[zip(range(self.ntopics), range(self.ntopics))] += self.l2v
        self.V[item_index] = np.linalg.solve(m, self.alpha*np.sum(Um, axis=0))


class _function_wrapper(object):

    def __init__(self, target, attr, *args, **kwargs):
        self.target = target
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def __call__(self, v):
        return getattr(self.target, self.attr)(v, *self.args, **self.kwargs)
