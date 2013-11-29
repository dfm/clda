#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LDA"]

import numpy as np
import scipy.special as sp
from scipy.misc import logsumexp


def dirichlet_expectation(g):
    if len(g.shape) == 1:
        return sp.psi(g) - sp.psi(np.sum(g))
    return sp.psi(g) - sp.psi(np.sum(g, axis=-1))[:, None]


class LDA(object):

    def __init__(self, ntopics, nvocab, alpha, eta, tau=1024., kappa=0.5):
        self.ntopics = ntopics
        self.nvocab = nvocab
        self.alpha = alpha
        self.eta = eta
        self.tau = tau
        self.kappa = kappa

        # Randomly initialize the lambda matrix.
        self.lam = np.random.gamma(100., 1./100., (self.ntopics, nvocab))
        self.elogbeta = dirichlet_expectation(self.lam)
        self.expelogbeta = np.exp(self.elogbeta)

    def sample(self, size, rate=100):
        # Generate topic distributions.
        beta = np.random.dirichlet(self.eta*np.ones(self.nvocab),
                                   self.ntopics)
        theta = np.random.dirichlet(self.alpha*np.ones(self.ntopics), size)
        nwords = np.random.poisson(rate, size)

        # Generate documents.
        docs = []
        for d, (n, th) in enumerate(zip(nwords, theta)):
            zs = np.random.multinomial(n, th)
            docs.append([w for i, z in enumerate(zs)
                         for w, c in enumerate(np.random.multinomial(z,
                                                                     beta[i]))
                         for j in range(c)])

        return docs, theta

    def infer(self, document, stats=None, maxiter=500, tol=1e-6):
        gamma = np.random.gamma(100., 1./100., self.ntopics)

        # Pre-compute the beta expectation.
        expelogbeta = self.expelogbeta[:, document]

        # Iterate between phi and gamma updates.
        delta = None
        for i in range(maxiter):
            # Update phi stats.
            expelogth = np.exp(dirichlet_expectation(gamma))
            norm = np.dot(expelogth, expelogbeta)

            # Check for convergence.
            if i > 0 and delta < tol:
                break

            # Update the vectors.
            gamma_new = (self.alpha
                         + expelogth * np.sum(expelogbeta/norm[None, :], 1))
            delta = np.mean(np.abs(gamma_new - gamma))
            gamma = np.array(gamma_new)

        # Update the sufficient stats.
        if stats is not None:
            stats[:, document] += (expelogbeta * expelogth[:, None]) / norm

        return gamma

    def rate(self, t):
        return (self.tau + t) ** -self.kappa

    def em(self, corpus, ndocs=None, batch=1024, maxiter=500, tol=1e-6):
        if ndocs is None:
            ndocs = len(corpus)

        t = 0
        documents = []
        elbo = 0.0
        for document in corpus:
            # Accumulate documents in the batch until the batch size is
            # reached.
            documents.append(document)
            if len(documents) < batch:
                continue

            # Run the expectation step.
            lam_new = np.zeros_like(self.lam)
            gammas = [self.infer(d, stats=lam_new, maxiter=maxiter, tol=tol)
                      for d in documents]
            lam_new = self.eta + ndocs * lam_new / batch

            # Estimate the ELBO.
            elbo += self.elbo(documents, gammas=gammas, ndocs=ndocs)

            # Do the stochastic update.
            rho = self.rate(t)
            self.lam = (1-rho)*self.lam + rho*lam_new
            self.elogbeta = dirichlet_expectation(self.lam)
            self.expelogbeta = np.exp(self.elogbeta)

            # Act as an iterator by yielding the evidence lower bound.
            yield elbo / (t+1)

            # Finalize.
            t += 1
            documents = []

    def elbo(self, documents, gammas=None, ndocs=None, maxiter=500, tol=1e-6):
        if ndocs is None:
            ndocs = len(documents)

        # Run the inference if the variational parameters are not provided.
        if gammas is None:
            gammas = [self.infer(d, maxiter=maxiter, tol=tol)
                      for d in documents]

        # Loop over documents and compute the approximate bound.
        elbo = sp.gammaln(self.nvocab*self.eta)
        elbo -= self.nvocab*sp.gammaln(self.eta)
        elbo += np.sum(sp.gammaln(self.lam))
        elbo -= np.sum(sp.gammaln(np.sum(self.lam, axis=1)))
        elbo += np.sum((self.eta - self.lam) * self.elogbeta)
        elbo /= ndocs

        elbo += sp.gammaln(self.ntopics*self.alpha)
        elbo -= self.ntopics*sp.gammaln(self.alpha)
        elbo *= len(documents)

        elbo += np.sum(np.sum(sp.gammaln(gammas), axis=1)
                       - sp.gammaln(np.sum(gammas, axis=1)))

        for doc, gamma in zip(documents, gammas):
            elogbeta = self.elogbeta[:, doc]
            elogth = dirichlet_expectation(gamma)
            lnphi = elogth[:, None] + elogbeta
            lnnorm = logsumexp(lnphi, axis=0)
            elbo += np.sum(np.exp(lnphi-lnnorm)*(elogth[:, None] + elogbeta
                                                 - lnphi + lnnorm))
            elbo += np.sum((self.alpha-gamma)*elogth)

        return elbo


class TestCorpus(object):

    def __init__(self, documents):
        self.ndocs = len(documents)
        self.documents = documents

    def __len__(self):
        return self.ndocs

    def __iter__(self):
        return self

    def next(self):
        return self.documents[np.random.randint(self.ndocs)]


if __name__ == "__main__":
    # Test stats.
    ntopics = 10
    nvocab = 5000

    model = LDA(ntopics, nvocab, 0.01, 0.01)
    print("Generating corpus...")
    corpus, true_theta = model.sample(500)
    print("Done.")

    for elbo in model.em(TestCorpus(corpus)):
        print(elbo)
