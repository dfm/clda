#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

pgm = daft.PGM([6, 3.5], origin=[0.5, -1.75])

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

x = 0.45
y = 0.5
pgm.add_plate(daft.Plate([1.4+x, -2.1+y, 2.2, 1.7], r"$D$"))
pgm.add_plate(daft.Plate([2.5+x, -2+y, 1, 1.5], r"$N_d$"))
pgm.add_plate(daft.Plate([3.7+x, -2+y, 1, 1.5], r"$K$",
                         position="bottom right"))

pgm.add_node(daft.Node("gamma", r"$\gamma_d$", 2+x, -1+y, fixed=True))
pgm.add_node(daft.Node("theta2", r"$\theta_d$", 2+x, -1.5+y))
pgm.add_edge("gamma", "theta2")

pgm.add_node(daft.Node("phi", r"$\phi_{d,n}$", 3+x, -1+y, fixed=True))
pgm.add_node(daft.Node("z2", r"$z_{d,n}$", 3+x, -1.5+y))
pgm.add_edge("phi", "z2")

pgm.add_node(daft.Node("lambda", r"$\lambda_{k}$", 4.2+x, -1+y, fixed=True))
pgm.add_node(daft.Node("beta2", r"$\beta_{k}$", 4.2+x, -1.5+y))
pgm.add_edge("lambda", "beta2")

pgm.render()
pgm.figure.savefig("lda.pdf")
