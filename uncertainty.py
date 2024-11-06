import os
import numpy as np
import pandas as pd


def affine_mat(pds, wds, t):
    mat_size = (pds.n_bus * 2 + wds.n_tanks) * t
    mat = np.zeros((mat_size, mat_size))

    load_std = pd.read_csv(os.path.join(pds.data_folder, 'dem_active_power_std.csv'), index_col=0) * pds.to_pu
    for i, s in enumerate(load_std.columns):
        block = np.zeros((pds.n_bus, pds.n_bus))
        np.fill_diagonal(block, load_std[s].values)
        mat[i * pds.n_bus: (i + 1) * pds.n_bus, i * pds.n_bus: (i + 1) * pds.n_bus] = block @ block

    pv_std = (pd.read_csv(os.path.join(pds.data_folder, 'pv_std.csv'), index_col=0).T * pds.bus['max_pv_pu'].values).T
    for i, s in enumerate(pv_std.columns):
        block = np.zeros((pds.n_bus, pds.n_bus))
        np.fill_diagonal(block, pv_std[s].values)
        mat[pds.n_bus * t + i * pds.n_bus: pds.n_bus * t + (i + 1) * pds.n_bus,
        pds.n_bus * t + i * pds.n_bus: pds.n_bus * t + (i + 1) * pds.n_bus] = block @ block

    dem_std = pd.read_csv(os.path.join(wds.data_folder, 'demands_std.csv'), index_col=0)
    for i, s in enumerate(dem_std.columns):
        block = np.zeros((t, t))
        np.fill_diagonal(block, dem_std[s].values)
        mat[pds.n_bus * t * 2 + i * t: pds.n_bus * t * 2 + (i + 1) * t,
        pds.n_bus * t * 2 + i * t: pds.n_bus * t * 2 + (i + 1) * t] = block @ block

    if not is_pd(mat):
        mat = nearest_positive_defined(mat)

    delta = np.linalg.cholesky(mat)

    certain_load_idx = np.where(load_std.values.flatten('F') == 0)[0]
    certain_pv_idx = np.where(pv_std.values.flatten('F') == 0)[0]
    certain_pv_idx = [pds.n_bus * t + _ for _ in certain_pv_idx]

    n_rows = delta.shape[0]
    mask = np.zeros(n_rows, dtype=bool)
    mask[certain_load_idx] = True
    mask[certain_pv_idx] = True
    delta[mask] = 0
    return delta


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