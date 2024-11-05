import os
import numpy as np
import pandas as pd

POWER_UNITS = {"w": 1, "kw": 1000, "mw": 10 ** 6}


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def consecutive_elements_mat(t):
    """
    generate a matrix that can be multiplied by decision variables to obtain differences between consecutive variables
    if x is of shape n x t and A is the returned matrix: diff = x @ A
    if x is of shape t x n and A is the returned matrix: diff = A.T @ x (A need to be transposed)
    :param t: size of x - number of time steps
    :return: consecutive diff matrix (numpy.array)
    """
    a = np.zeros((t, t))
    np.fill_diagonal(a, -1)
    a[np.arange(1, t), np.arange(t - 1)] = 1
    return a


def connectivity_mat(edges_data: pd.DataFrame, from_col: str = 'from', to_col: str = 'to', direction='', param=''):
    n_edges = len(edges_data)
    n_nodes = pd.concat([edges_data[from_col], edges_data[to_col]]).nunique()

    mat = np.zeros((n_nodes, n_edges))
    mat[edges_data.loc[:, from_col], np.arange(n_edges)] = -1
    mat[edges_data.loc[:, to_col], np.arange(n_edges)] = 1

    if direction == 'in':
        mat[mat == -1] = 0
    if direction == 'out':
        mat[mat == 1] = 0

    if param:
        # row-wise multiplication
        mat = mat * edges_data[param].values
    return mat


def get_mat_for_type(data: pd.DataFrame, category_data: pd.DataFrame, inverse=False):
    """
    generate a matrix that can be multiplied by nodes / edges vector to get nodes / edges of certain type
    returns a NxN matrix that is based on an eye matrix where only nodes / edges from the requested type are 1
    N is the number of nodes / edges
    inverse - to return all nodes /edges beside the input type

    data: pd.DataFrame - probably one of: wds.nodes, wds.pipes, pds.bus, pds.lines
    """
    idx = np.where(data.index.isin(category_data.index.to_list()), 1, 0)

    if inverse:
        # to get a matrix of all types but the input one
        idx = np.logical_not(idx)

    mat = idx * np.eye(len(data))
    return mat


def quad_to_piecewise(a, b, c, p_min, p_max, num_segments):
    """
    Converts a quadratic generator cost function to a two segments piecewise linear approximation.
    Args:
    a (float): Coefficient of P^2 term in the quadratic cost function.
    b (float): Coefficient of P term in the quadratic cost function.
    c (float): Constant term in the quadratic cost function.
    p_max (float): Maximum active power output.
    p_max (float): Maximum active power output.
    num_segments (int, optional): Number of linear segments (default=2).
    Returns: A list of tuples, where each tuple contains the x, y of the piecewise linear breakpoints
    """
    def quad_cost(a, b, c, x):
        return a * x ** 2 + b * x + c

    x_breakpoints = [p_min + (p_max - p_min) * i / num_segments for i in range(num_segments + 1)]
    y_breakpoints = [quad_cost(a, b, c, x) for x in x_breakpoints]

    return tuple(zip(x_breakpoints, y_breakpoints))