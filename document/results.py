import numpy as np
import matplotlib.pyplot as pl

N = np.arange(50, 501, 50)
data = np.loadtxt("results.txt")

fig = pl.figure(figsize=(5, 5))
fig.subplots_adjust(top=0.975, right=0.96, bottom=0.12, left=0.135)
for d, s, l in zip(data, ["o", "s", "*", "d", "^"],
                   ["random", "tf-idf", "LDA", "ICF", "CLDA"]):
    pl.plot(N, d, "-{0}".format(s), lw=2, ms=10, label=l)

pl.locator_params(nbins=6)
# pl.legend()
pl.xlabel("number of recommended articles")
pl.ylabel("held-out recall")
pl.ylim(0, 0.85)
pl.xlim(50, 500)
pl.savefig("results.pdf")
