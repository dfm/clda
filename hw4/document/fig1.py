#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([6.5, 2], origin=[-1, 0], node_ec="none", directed=False)

pgm.add_node(daft.Node("null", r"", -0.5, 1.25))
pgm.add_node(daft.Node("la", r"la$_1$", -0.5, 0.5))
pgm.add_node(daft.Node("blue", r"blue$_1$", 0.5, 1.5))
pgm.add_node(daft.Node("maison", r"maison$_2$", 0.5, 0.5))
pgm.add_node(daft.Node("house", r"house$_2$", 1.5, 1.5))
pgm.add_node(daft.Node("bleu", r"bleu$_3$", 1.5, 0.5))
pgm.add_edge("la", "null")
pgm.add_edge("blue", "maison")
pgm.add_edge("house", "bleu")

pgm.add_node(daft.Node("null2", r"", 3, 1.25))
pgm.add_node(daft.Node("la2", r"la$_1$", 3, 0.5))
pgm.add_node(daft.Node("blue2", r"blue$_1$", 4, 1.5))
pgm.add_node(daft.Node("maison2", r"maison$_2$", 4, 0.5))
pgm.add_node(daft.Node("house2", r"house$_2$", 5, 1.5))
pgm.add_node(daft.Node("bleu2", r"bleu$_3$", 5, 0.5))
pgm.add_edge("la2", "null2")
pgm.add_edge("house2", "maison2")
pgm.add_edge("blue2", "bleu2")

pgm.render()
pgm.figure.savefig("fig1.pdf")
