import numpy as np
import matplotlib.pyplot as pl

experiments = np.array([
    (1, float("inf"), 0, 57.1369509044, 65.6740270118),
    (2, 1, 0, 56.6786009364, 64.4278801124),
    (3, 2, 0, 58.1456404588, 65.4130052724),
    (4, 2, 1, 63.5070451021, 71.4927495024),
    (5, 2, 2, 63.9447109739, 73.0497122913),
    (6, 3, 2, 64.402574762, 72.9830508475),
])

pl.plot(experiments[:, 0], experiments[:, 3], "-k")
pl.plot(experiments[:, 0], experiments[:, 3], "ok")
pl.xlim(0.5, len(experiments)+0.5)
pl.gca().set_xticks(range(1, len(experiments)+1))
pl.gca().set_xticklabels([chr(ord("A") + i) for i in range(len(experiments))])
pl.xlabel("experiment")
pl.ylabel("F1 score")
pl.savefig("out-of-domain.pdf")

pl.clf()
pl.plot(experiments[:, 0], experiments[:, 4], "-k")
pl.plot(experiments[:, 0], experiments[:, 4], "ok")
pl.xlim(0.5, len(experiments)+0.5)
pl.gca().set_xticks(range(1, len(experiments)+1))
pl.gca().set_xticklabels([chr(ord("A") + i) for i in range(len(experiments))])
pl.xlabel("experiment")
pl.ylabel("F1 score")
pl.savefig("in-domain.pdf")
