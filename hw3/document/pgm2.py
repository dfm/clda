#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([6, 2], origin=[0, 0], node_unit=1.3)

pgm.add_node(daft.Node("start", r"$\left < s, s \right >$", 0.5, 1.5))
pgm.add_node(daft.Node("t1", r"$\left < s, t_1 \right >$", 1.5, 1.5))
pgm.add_node(daft.Node("t2", r"$\left < t_1, t_2 \right >$", 2.5, 1.5))
pgm.add_node(daft.Node("ldots", r"\ldots", 3.5, 1.5,
                       plot_params={"ec": "none"}))
pgm.add_node(daft.Node("tn", r"$\left < t_n, e \right >$", 4.5, 1.5))
pgm.add_node(daft.Node("end", r"$\left < e, e \right >$", 5.5, 1.5))

pgm.add_node(daft.Node("w1", r"$w_1$", 1.5, 0.5, observed=True))
pgm.add_node(daft.Node("w2", r"$w_2$", 2.5, 0.5, observed=True))
pgm.add_node(daft.Node("wn", r"$w_n$", 4.5, 0.5, observed=True))

pgm.add_edge("start", "t1")
pgm.add_edge("t1", "t2")
pgm.add_edge("t2", "ldots")
pgm.add_edge("ldots", "tn")
pgm.add_edge("tn", "end")

pgm.add_edge("start", "w1")
pgm.add_edge("t1", "w1")
pgm.add_edge("t1", "w2")
pgm.add_edge("t2", "w2")
pgm.add_edge("ldots", "wn")
pgm.add_edge("tn", "wn")

pgm.render()
pgm.figure.savefig("pgm2.pdf")
