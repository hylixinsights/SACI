"""
scoring.py - Statistical tests for multimodality detection

Implements:
    - Hartigan's Dip Test (non-parametric, tests unimodality)
    - Gaussian Mixture Model k=1 vs k=2 with BIC
    - Composite multimodal score
"""

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Hartigan's Dip Test (diptest C implementation)
# ---------------------------------------------------------------------------

import diptest as dt

def dip_test(values: np.ndarray) -> tuple[float, float]:
    """
    Hartigan's Dip Test for unimodality.

    H0: distribution is unimodal
    Small p-value → reject unimodality → evidence of multimodality

    Parameters
    ----------
    values : np.ndarray
        1D array of expression values (zeros already removed).

    Returns
    -------
    dip : float
        Dip statistic. Larger = more multimodal.
    p_value : float
        P-value for the test. Small p → multimodal.
    """
    dip, pval = dt.diptest(np.asarray(values, dtype=float))
    return float(dip), float(pval)


# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------

def fit_gmm(values: np.ndarray) -> dict:
    """
    Fit GMM with k=1 and k=2 components, compare via BIC.

    Parameters
    ----------
    values : np.ndarray
        1D array of non-zero log-normalized expression values.

    Returns
    -------
    dict with keys:
        delta_bic       : BIC(k=1) - BIC(k=2). Positive = k=2 is better.
        peak_separation : Cohen's d between the two GMM component means.
                          0.0 if k=2 not favored.
        means           : [mean1, mean2] of the two components (or [mean] for k=1)
        weights         : mixture weights for k=2
        bic_k1          : raw BIC for k=1
        bic_k2          : raw BIC for k=2
        best_k          : 1 or 2
    """
    X = values.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=42, max_iter=200)
    gmm2 = GaussianMixture(n_components=2, random_state=42, max_iter=200)

    gmm1.fit(X)
    gmm2.fit(X)

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)
    delta_bic = bic1 - bic2  # positive = k=2 is better

    best_k = 2 if delta_bic > 0 else 1

    if best_k == 2:
        means = gmm2.means_.flatten()
        weights = gmm2.weights_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())

        # Sort components by mean
        order = np.argsort(means)
        means = means[order]
        weights = weights[order]
        stds = stds[order]

        # Cohen's d between the two peaks
        pooled_std = np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2)
        peak_sep = abs(means[1] - means[0]) / (pooled_std + 1e-8)
    else:
        means = gmm1.means_.flatten()
        weights = gmm1.weights_.flatten()
        peak_sep = 0.0

    return {
        "delta_bic": float(delta_bic),
        "peak_separation": float(peak_sep),
        "means": means.tolist(),
        "weights": weights.tolist(),
        "bic_k1": float(bic1),
        "bic_k2": float(bic2),
        "best_k": best_k,
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def multimodal_score(dip_pvalue: float, delta_bic: float, peak_separation: float) -> float:
    """
    Composite multimodality score for a single gene.

    score = (1 - dip_pvalue) * max(delta_BIC, 0) * peak_separation

    Properties:
        - 0 if dip_pvalue = 1 (perfectly unimodal by dip test)
        - 0 if delta_BIC <= 0 (k=1 fits as well or better than k=2)
        - 0 if peak_separation = 0 (two peaks completely overlapping)
        - Higher score = clearer bimodal structure

    Parameters
    ----------
    dip_pvalue : float
        P-value from Hartigan's dip test.
    delta_bic : float
        BIC(k=1) - BIC(k=2). Positive means k=2 is preferred.
    peak_separation : float
        Cohen's d between the two GMM components.

    Returns
    -------
    float
        Composite score. Higher is better.
    """
    return (1.0 - dip_pvalue) * max(delta_bic, 0.0) * peak_separation
