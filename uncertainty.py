import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import graphs
from pds import PDS
from wds import WDS


def affine_mat(pds, wds, t):
    mat_size = (pds.n_bus * 2 + wds.n_tanks) * t
    cov = np.zeros((mat_size, mat_size))
    corr = cyclic_periodic_corr(A=0.1, alpha=0.1, t=t)

    load_std = pds.dem_active_std.iloc[:, :t]
    pv_std = pds.pv_std[:, :t].T
    dem_std = wds.demands_std.iloc[:t, :]

    for _ in range(pds.n_bus):
        block = np.zeros((t, t))
        np.fill_diagonal(block, load_std.iloc[_, :])
        cov[_ * t: (_ + 1) * t, _ * t: (_ + 1) * t] = block @ corr @ block

        block = np.zeros((t, t))
        np.fill_diagonal(block, pv_std[:, _])
        cov[pds.n_bus * t + _ * t: pds.n_bus * t + (_ + 1) * t, pds.n_bus * t + _ * t: pds.n_bus * t + (_ + 1) * t] = block @ corr @ block

    for _ in range(wds.n_tanks):
        block = np.zeros((t, t))
        np.fill_diagonal(block, dem_std.iloc[:, _])
        cov[pds.n_bus * t * 2 + _ * t: pds.n_bus * t * 2 + (_ + 1) * t,
            pds.n_bus * t * 2 + _ * t: pds.n_bus * t * 2 + (_ + 1) * t] = block @ corr @ block

    delta = cholesky_with_zeros(cov)
    return cov, delta


def affine_mat_ewri(mu, t, std=0.05):
    cov = np.zeros((len(mu), len(mu)))

    sigma_loads = np.zeros((t, t))
    loads = mu[[_ * 2 + 1 for _ in range(t)]]
    np.fill_diagonal(sigma_loads, loads * std)

    cov_water = np.zeros((t, t))
    dem = mu[[2 * t + _ for _ in range(t)]]
    np.fill_diagonal(cov_water, dem * std)

    corr = cyclic_periodic_corr(A=0.1, alpha=0.1, t=t)
    cov_loads = sigma_loads @ corr @ sigma_loads
    cov_water_dem = cov_water @ corr @ cov_water

    for i in range(t):
        for j in range(t):
            cov[2 * i + 1, 2 * j + 1] = cov_loads[i, j]

    cov[-t:, -t:] = cov_water_dem
    if not is_pd(cov):
        cov = nearest_positive_defined(cov)
    delta = np.linalg.cholesky(cov)

    # get rows with no uncertainty
    idx = np.where(mu == 0)[0]
    delta[:, idx] = 0
    cov[:, idx] = 0
    return cov, delta


def nearest_positive_defined(mat):
    """
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    b = (mat + mat.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))
    mat2 = (b + h) / 2
    mat3 = (mat2 + mat2.T) / 2
    if is_pd(mat3):
        return mat3

    spacing = np.spacing(np.linalg.norm(mat))
    k = 1
    while not is_pd(mat3):
        mineig = np.min(np.real(np.linalg.eigvals(mat3)))
        mat3 += np.eye(mat.shape[0]) * (-mineig * k**2 + spacing)
        k += 1

    return mat3


def is_pd(mat):
    """
    Returns true when input is positive-definite, via Cholesky
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    try:
        _ = np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


def cyclic_periodic_corr(A, alpha, t=24):
    """
    corr(x_i, x_i+j) = A * exp(-alpha * j) * cos(2 * pi * j / T)
    """
    mat = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            if i == j:
                mat[i, j] = 1
            else:
                lag = abs(i - j)
                mat[i, j] = A * np.exp(-alpha * lag) * np.cos(2 * np.pi * lag / 24)  # T=24 due to daily cycles
    return mat


def draw_multivariate(mu, cov, n):
    # Create a random number generator
    rng = np.random.default_rng(seed=42)

    # Identify indices where mu is not zero
    non_zero_indices = np.nonzero(mu)[0]

    # Extract the non-zero entries from mu and the corresponding rows and columns from cov
    mu_non_zero = mu[non_zero_indices]

    cov_non_zero = cov[np.ix_(non_zero_indices, non_zero_indices)]

    # Draw samples from the modified distribution
    samples_non_zero = rng.multivariate_normal(mu_non_zero, cov_non_zero, size=n).T

    # Create an output array filled with zeros of the original shape
    output_samples = np.zeros((len(mu), n))

    # Insert the non-zero samples back into the correct positions
    output_samples[non_zero_indices] = samples_non_zero

    return output_samples


def cholesky_with_zeros(cov):
    # Identify indices with non-zero variance (non-zero diagonal elements in the covariance matrix)
    non_zero_indices = np.nonzero(np.diag(cov))[0]

    # Form the reduced covariance matrix
    reduced_cov = cov[np.ix_(non_zero_indices, non_zero_indices)]

    # Perform Cholesky decomposition on the reduced matrix
    L_reduced = np.linalg.cholesky(reduced_cov)

    # Create a full L matrix with original dimensions, initialized to zero
    L_full = np.zeros_like(cov)

    # Place the Cholesky factor back into the correct positions
    L_full[np.ix_(non_zero_indices, non_zero_indices)] = L_reduced

    return L_full



