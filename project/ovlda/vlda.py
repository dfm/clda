#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
import scipy.special as sp


def vlda_inference(alpha, beta, document, gamma=None, maxiter=500, tol=1e-6):
    ntopics, nvocab = beta.shape
    assert ntopics == len(alpha), "Dimension mismatch"
    if gamma is None:
        gamma = np.ones_like(alpha)
    for i in range(maxiter):
        phi = np.exp(sp.psi(gamma))[:, None] * beta[:, document]
        phi /= phi.sum(axis=0)[None, :]
        gamma_new = alpha + phi.sum(axis=1)
        delta = np.mean(np.abs(gamma_new - gamma))
        gamma = np.array(gamma_new)
        if delta < tol:
            break
    stats = np.zeros_like(beta)
    stats[:, document] += phi
    return gamma, stats


def vlda_em(nvocab, alpha, corpus, eta=0.01, maxiter=50, tol=1e-6,
            maxinf=500, tolinf=1e-6):
    ntopics = len(alpha)
    beta = np.random.rand(ntopics*nvocab).reshape((ntopics, nvocab))
    beta /= beta.sum(axis=1)[:, None]
    gamma = np.ones_like(alpha)

    # Expectation step.
    for i in range(maxiter):
        beta_new = eta + np.zeros_like(beta)
        for document in corpus:
            gamma, stats = vlda_inference(alpha, beta, document, gamma=gamma,
                                          maxiter=maxinf, tol=tolinf)
            beta_new += stats
        delta = np.mean(np.abs(beta_new - beta))
        beta = np.array(beta_new)
        if delta < tol:
            break

    return beta


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

    # Run EM.
    new_beta = vlda_em(nvocab, alpha, corpus)
    print(np.mean(np.abs(beta - new_beta)))
