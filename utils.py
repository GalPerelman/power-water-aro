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


def get_relative_indices(a, b):
    """
    This function help to find the relative locations of the indep and dep elements
    the variables vectors is structured as follows:
    Construct the bus power balance matrix for the case of equality constraints
    variables order (matrix columns), for t in range(self.t):

    [θ0, θ1, ... θN | Xg0, Xg1, ... XgN | bat0C, Xbat1C, ... batNC | bat0D, Xbat1D, ... batND |
    comb0, comb1 ... combN | desal0, desal1 ... deslaN | tank0, tank1 ... tankN]

    In the end are the piecewise linear variables:
    [pw_g0_t0, pw_g1_t0 ... pw_G_t0, ... pw_g0_T, pw_g1_T ... pw_G_T]

    Because the piecewise linear variables are located at the end of the vector
    When we extract the indep and dep variables of a timestep (for example t=0) we will get:
    indep = [0, 1, 5, 6, 100], dep = [2, 3, 4]
    The piecewise linear variable is indexed based on its location in the entire vector
    and not based on the current time step

    p1 and p2 are auxiliary matrices that helps build the decision variables vector based on the independent variables
    For constructing p1 and p2 of a single time step we need to identify indep and dep indexes
    relative to the time stpe variables
    """
    # Combine and sort all unique values
    all_values = sorted(set(a + b))

    # Create a mapping from value to index
    value_to_index = {val: i for i, val in enumerate(all_values)}

    # Map each list to their indices
    a_iloc = [value_to_index[x] for x in a]
    b_iloc = [value_to_index[x] for x in b]

    return a_iloc, b_iloc


def single_var_non_anticipative_mat(t, lags, lat):
    """
    :param k:       block size - number of LDRs
    :param t:       number of optional time steps - optimization horizon
    :param lags:    number of lags backwards dependencies of the LDR
    :param lat:     number of latency time steps
    :return:        block matrix
    """
    total_size = t
    mat = np.zeros((total_size, total_size))

    # Fill the matrix with identity matrices according to `lags`
    for diag in range(1, lags + 1):
        for block in range(diag, t):
            start_row = block
            end_row = start_row + 1
            start_col = (block - diag)
            end_col = start_col + 1
            mat[start_row:end_row, start_col:end_col] = 1

    # Apply `lat` to zero out the specified diagonals
    for diag in range(1, lat + 1):
        for block in range(diag, t):
            start_row = block
            end_row = start_row + 1
            start_col = (block - diag)
            end_col = start_col + 1
            mat[start_row:end_row, start_col:end_col] = 0

    return mat