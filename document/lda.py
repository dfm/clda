#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

pgm = daft.PGM([5.6, 1.4], origin=[0.75, 0.3])

pgm.add_plate(daft.Plate([1.4, 0.4, 3.1, 1.2], r"$D$"))
pgm.add_plate(daft.Plate([2.5, 0.5, 1.95, 1], r"$N_d$"))
pgm.add_plate(daft.Plate([4.6, 0.5, 1, 1], r"$K$", position="bottom right"))

pgm.add_node(daft.Node("alpha", r"$\alpha$", 1, 1, fixed=True))
pgm.add_node(daft.Node("theta", r"$\theta_d$", 2, 1))
pgm.add_node(daft.Node("z", r"$z_{d,n}$", 3, 1))
pgm.add_node(daft.Node("w", r"$w_{d,n}$", 4, 1, observed=True))

pgm.add_node(daft.Node("beta", r"$\beta_{k}$", 5.1, 1))
pgm.add_node(daft.Node("eta", r"$\eta$", 6.1, 1, fixed=True))

pgm.add_edge("alpha", "theta")
pgm.add_edge("theta", "z")
pgm.add_edge("z", "w")

pgm.add_edge("eta", "beta")
pgm.add_edge("beta", "w")

pgm.render()
pgm.figure.savefig("lda.pdf")
