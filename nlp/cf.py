#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import sqlite3
import numpy as np

with sqlite3.connect("data/abstracts.db") as connection:
    c = connection.cursor()
    c.execute("""SELECT user_id FROM (SELECT user_id, COUNT(user_id) AS c
                                      FROM activity GROUP BY user_id)
                 WHERE c > 10""")
    users = [u[0] for u in c]

    c.execute("""SELECT arxiv_id
                 FROM activity GROUP BY arxiv_id""")
    articles = [a[0] for a in c]

user_map = {}
for u in users:
    user_map[u] = len(user_map)
nusers = len(user_map)
print("{0} users".format(nusers))

article_map = {}
for a in articles:
    article_map[a] = len(article_map)
narticles = len(article_map)
print("{0} articles".format(narticles))

# Get the data.
with sqlite3.connect("data/abstracts.db") as connection:
    c = connection.cursor()
    data = []
    for u in users:
        c.execute("SELECT arxiv_id FROM activity WHERE user_id=?", (u, ))
        data.append([article_map[a[0]] for a in c])

ntopics = 50
V = np.random.randn(narticles, ntopics)
U = np.empty((nusers, ntopics))

a = 40.0
l2 = 0.1

for i, r in enumerate(data):
    print(i, len(r))
    VTV = np.dot(V.T, V)
    Vm = V[r]
    VmTVm = np.dot(Vm.T, Vm)
    m = VTV + a * VmTVm
    m[zip(range(ntopics), range(ntopics))] += l2
    U[i] = np.linalg.solve(m, a * np.sum(Vm, axis=0))
