#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
import scipy.special as sp


def dirichlet_expectation(g):
    return sp.psi(g) - sp.psi(np.sum(g, axis=-1))


def vlda_inference(alpha, beta, document, gamma=None, maxiter=500, tol=1e-6):
    ntopics, nvocab = beta.shape
    assert ntopics == len(alpha), "Dimension mismatch"
    gamma = np.ones_like(alpha)
    for i in range(maxiter):
        phi = np.exp(sp.psi(gamma))[:, None] * beta[:, document]
        phi /= phi.sum(axis=0)[None, :]
        gamma_new = alpha + phi.sum(axis=1)
        delta = np.mean(np.abs(gamma_new - gamma))
        gamma = np.array(gamma_new)
        if delta < tol:
            break
    return gamma, phi


def vlda_em(nvocab, alpha, corpus, eta=0.01, maxiter=50, tol=1e-4,
            maxinf=500, tolinf=1e-6):
    ntopics = len(alpha)
    beta = np.random.rand(ntopics*nvocab).reshape((ntopics, nvocab))
    beta /= beta.sum(axis=1)[:, None]

    # Expectation step.
    norm = sum([len(d) for d in corpus])
    old_perplex = None
    for i in range(maxiter):
        beta_new = eta + np.zeros_like(beta)
        perplex = 0.0
        for document in corpus:
            gamma, stats = vlda_inference(alpha, beta, document,
                                          maxiter=maxinf, tol=tolinf)

            # Maximization update on beta.
            beta_new[:, document] += stats

            perplex += approx_perplexity(alpha, beta, document, gamma=gamma,
                                         stats=stats)

        # Update the beta matrix.
        beta = np.array(beta_new)

        # Check for convergence.
        perplex = np.exp(perplex/norm)
        print("perplexity = {0}".format(perplex))
        if i > 0 and np.abs(perplex - old_perplex) < tol:
            break
        old_perplex = perplex

    return beta


def ovlda_em(nvocab, alpha, corpus, eta=0.01, maxiter=50, tol=1e-3,
             maxinf=500, tolinf=1e-6):
    ntopics = len(alpha)
    beta = np.random.rand(ntopics*nvocab).reshape((ntopics, nvocab))
    beta /= beta.sum(axis=1)[:, None]


def approx_perplexity(alpha, beta, document, gamma=None, stats=None,
                      maxiter=500, tol=1e-6):
    if gamma is None or stats is None:
        gamma, stats = vlda_inference(alpha, beta, document, maxiter=maxiter,
                                      tol=tol)

    psi = sp.psi(gamma) - sp.psi(np.sum(gamma))
    lnlike = sp.gammaln(np.sum(alpha)) - np.sum(sp.gammaln(alpha))
    lnlike += np.sum((alpha - 1) * psi)
    lnlike += np.sum(stats*psi[:, None])
    lnlike += np.sum(stats*np.log(beta[:, document]))

    lnlike -= sp.gammaln(np.sum(gamma)) - np.sum(sp.gammaln(gamma))
    lnlike -= np.sum((gamma - 1) * psi)
    lnlike -= np.sum(stats*np.log(stats))

    return -lnlike


def generate_document(alpha, beta, nwords):
    # Generate a test document.
    theta = np.array([np.random.gamma(a) for a in alpha])
    theta /= theta.sum()
    z = np.array([int(np.arange(nvocab)[i])
                 for i in np.array(np.random.multinomial(1, theta, nwords),
                                   dtype=bool)])
    document = np.array([np.argmax(np.random.multinomial(1, d))
                         for d in beta[z, :]])

    return document


if __name__ == "__main__":
    # Test stats.
    ntopics = 10
    nvocab = 5000

    # Set up the test distribution.
    alpha = np.random.rand(ntopics)
    alpha /= alpha.sum()
    beta = np.random.rand(ntopics*nvocab).reshape((ntopics, nvocab))
    beta /= beta.sum(axis=1)[:, None]

    # Generate a corpus.
    corpus = [generate_document(alpha, beta, np.random.poisson(100))
              for i in range(100)]

    test_corpus = [generate_document(alpha, beta, np.random.poisson(100))
                   for i in range(100)]

    perplex = sum([approx_perplexity(alpha, beta, d) for d in test_corpus])
    perplex = np.exp(perplex / sum([len(d) for d in test_corpus]))
    print("True test perplexity = {0}".format(perplex))

    # Run EM.
    new_beta = vlda_em(nvocab, alpha, corpus)

    perplex = sum([approx_perplexity(alpha, new_beta, d) for d in test_corpus])
    perplex = np.exp(perplex / sum([len(d) for d in test_corpus]))
    print("Final test perplexity = {0}".format(perplex))
