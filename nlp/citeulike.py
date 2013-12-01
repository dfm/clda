#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["parse_linkouts"]

import re
import sqlite3
from collections import defaultdict


p = re.compile(r"([0-9]{4}\.[0-9]{4})"
               r"|([a-zA-Z\-\.]+?/[0-9]+)"
               r"|([a-zA-Z\-\.]+?\?[0-9]+)"
               r"|([a-zA-Z\-\.]+?/\?[0-9]+)"
               r"|([a-zA-Z\-\.]+?\?=[0-9]+)"
               r"|([a-zA-Z\-\.]+?%2F[0-9]+)"
               r"|([a-zA-Z\-\.]+?%2f[0-9]+)"
               r"|([a-zA-Z\-\.]+?\$/\$[0-9]+)"
               r"|([a-zA-Z\-\.]+?\.[0-9]+)"
               r"|([a-zA-Z\-\.]+?\?papernum=[0-9]+)")
p_pref = re.compile(r"([0-9]{4}\.[0-9]{4})")


def get_id(r):
    for i, el in enumerate(r):
        if len(el):
            if i == 0 or i == 1:
                return el
            elif i == 2:
                return "/".join(el.split("?"))
            elif i == 3:
                return "/".join(el.split("/?"))
            elif i == 4:
                return "/".join(el.split("?="))
            elif i == 5:
                return "/".join(el.split("%2F"))
            elif i == 6:
                return "/".join(el.split("%2f"))
            elif i == 7:
                return "/".join(el.split("$/$"))
            elif i == 8:
                return "/".join(["".join(el.split(".")[:-1]),
                                 el.split(".")[-1]])
            else:
                return "/".join(el.split("?papernum="))
    return None


def parse_linkouts(fn):
    final = dict()
    for line in open(fn):
        article_id = line.split("|")[0]

        results = p.findall(line)
        ids = map(get_id, results)
        if len(ids) == 0:
            continue

        if all([i == ids[0] for i in ids[1:]]):
            final[article_id] = ids[0]

        else:
            for i in ids:
                if len(p_pref.findall(i)):
                    final[article_id] = i
                    break
    return final


def parse_users(fn, linkouts):
    users = defaultdict(list)
    count = 0
    for line in open(fn):
        cols = line.split("|")
        linkout, user = cols[0], cols[1]
        if linkout in linkouts:
            count += 1
            users[user].append(linkouts[linkout])
    return users, count


if __name__ == "__main__":
    linkouts = parse_linkouts("data/arxiv")
    print("{0} unique articles.".format(len(set(linkouts.values()))))

    users, count = parse_users("data/current", linkouts)
    print("{0} unique users".format(len(users)))
    print("{0} user-article pairs".format(count))

    # Update the database.
    connection = sqlite3.connect("data/abstracts.db")
    c = connection.cursor()
    c.execute("PRAGMA foreign_keys = ON")
    # c.execute("DROP TABLE activity")
    c.execute("""
    CREATE TABLE IF NOT EXISTS activity(
        user_id  TEXT,
        arxiv_id INTEGER REFERENCES articles
    )""")

    for u, arxiv_ids in users.iteritems():
        arxiv_ids = [i if "/" not in i else
                     i.split("/")[0].split(".")[0]+"/"+i.split("/")[1]
                     for i in arxiv_ids]
        c.executemany("""INSERT INTO activity (user_id,arxiv_id)
                         VALUES (?,(SELECT id FROM articles WHERE arxiv_id=?))
                      """, [(u, i) for i in arxiv_ids])
        connection.commit()
