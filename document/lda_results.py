import numpy as np
import matplotlib.pyplot as pl

fig = pl.figure(figsize=(5, 5))
fig.subplots_adjust(top=0.975, right=0.96, bottom=0.13, left=0.155)
for k in [50, 100, 200, 300]:
    data = np.loadtxt("lda-results/k={0}.txt".format(k))
    t = data[:, 1]
    m = t < 1e4
    pl.plot(t[m], data[:, 2][m]*1e-3, "-o", lw=2, label="$K={0}$".format(k))
    pl.gca().set_xscale("log")
pl.legend()
pl.locator_params(axis="y", nbins=6)
pl.xlabel("walltime [seconds]")
pl.ylabel(r"held-out perplexity [$\times 10^3$]")
pl.ylim(1.4, 3.49)
pl.savefig("lda-results-k.pdf")

fig = pl.figure(figsize=(5, 5))
fig.subplots_adjust(top=0.975, right=0.96, bottom=0.13, left=0.155)
for k in [1024, 2048, 4096]:
    data = np.loadtxt("lda-results/s={0}.txt".format(k))
    t = data[:, 1]
    m = t < 1e4
    pl.plot(t[m], data[:, 2][m]*1e-3, "-o", lw=2, label="$S={0}$".format(k))
    pl.gca().set_xscale("log")
pl.legend()
pl.locator_params(axis="y", nbins=6)
pl.xlabel("walltime [seconds]")
pl.ylabel(r"held-out perplexity [$\times 10^3$]")
pl.ylim(1.66, 2.3)
pl.savefig("lda-results-s.pdf")
