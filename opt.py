import pickle
import time
import datetime
import numpy as np
import cvxpy as cp
import pandas as pd
import sympy as sp
import scipy
from scipy import sparse

import copy
import gurobipy as gp
import graphs
import utils
import uncertainty
from pds import PDS
from wds import WDS

np.set_printoptions(linewidth=10 ** 5)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)


class BaseOptModel:
    def __init__(self, pds_path, wds_path, t, omega=None, opt_method='aro', elimination_method='manual',
                 manual_indep_variables=None, pw_segments=None, n_bat_vars=2, solver_params=None, solver_display=False,
                 **kwargs):

        self.pds_path = pds_path
        self.wds_path = wds_path
        self.t = t
        self.omega = omega
        self.opt_method = opt_method
        self.elimination_method = elimination_method
        self.manual_indep_variables = manual_indep_variables
        self.pw_segments = pw_segments
        self.n_bat_vars = n_bat_vars
        self.solver_params = solver_params
        self.solver_display = solver_display

        self.EPSILON = 0.1
        self.piecewise_coef_scale = 1000  # to improve numerical stability

        self.pds = PDS(self.pds_path)
        self.wds = WDS(self.wds_path)
        self.wds_power_units_factor = (utils.POWER_UNITS[self.wds.input_power_units]
                                       / utils.POWER_UNITS[self.pds.input_power_units]) * self.pds.to_pu

        # for matrices dimensions
        self.n_bus, self.n_gen, self.n_bat = self.pds.n_bus, self.pds.n_generators, self.pds.n_batteries
        self.n_combs, self.n_desal, self.n_tanks = self.wds.n_combs, self.wds.n_desal, self.wds.n_tanks

        # total number of variables per time step (not including piecewise linear)
        # batteries have 2 variables (charging and discharging)
        self.n_pds = self.n_bus + self.n_gen + self.n_bat_vars * self.n_bat
        self.n_wds = self.n_combs + self.n_desal + self.n_tanks
        self.n_tot = self.n_bus + self.n_gen + self.n_bat_vars * self.n_bat + self.n_combs + self.n_desal + self.n_tanks
        self.w = 0 if self.pw_segments is None else self.pw_segments + 1  # number of segments edges

        # piecewise variables
        self.n_pw_vars = self.n_gen * self.t
        self.piecewise_costs = np.zeros(self.n_tot * self.t + self.n_pw_vars)

        # inequality constraints are saved in dicts {name: mat1, name: mat2..], [name: b1, name:b2..]
        self.ineq_mat, self.ineq_rhs = {}, {}
        # equality constraints only one mat and rhs for the variables decomposition
        self.eq_mat, self.eq_rhs = self.build_equality_constraints()

        self.build_inequality_constraints()

        self.formulation_start = None
        self.problem = None
        self.omega_param = None
        self.constraints = []
        self.solution_metrics = {}

        # time decomposition for ro and aro
        self.t_cols, self.t_eq_rows = self.get_entries_by_t()

    def manually_set_dep_columns(self):
        """
        not in use:
        experimental - check the impact of different independent sets
        to define independent variables - use the experiment yaml configurations
        """
        bus_angles = [_ for _ in range(self.n_bus)]
        generators = [self.n_bus + _ for _ in range(self.n_gen)]
        bat_charge = [self.n_bus + self.n_gen + _ for _ in range(self.n_bat)]
        if self.n_bat_vars == 2:
            bat_discharge = [self.n_bus + self.n_gen + self.n_bat + _ for _ in range(self.n_bat)]

        combs = [self.n_pds + _ for _ in range(self.n_combs)]
        desal = [self.n_pds + self.n_combs + _ for _ in range(self.n_desal)]
        tanks = [self.n_pds + self.n_combs + self.n_desal + _ for _ in range(self.n_tanks)]
        pw = [self.n_tot * self.t + _ for _ in range(self.n_gen)]
        indep_cols = [0] + [generators[0]] + [generators[2]] + bat_charge + combs + desal + pw  # ieee9 case study - 1
        # indep_cols = [0] + generators + combs + desal + pw  # ieee9 case study - 2
        # indep_cols = [0] + [generators[0]] + bat_charge + bat_discharge + combs + desal + pw  # ieee9 case study - 3
        # indep_cols = [0] + generators[:-1] + bat_charge + combs + desal + pw  # ewri case study
        dep_cols = [_ for _ in range(self.n_tot) if _ not in indep_cols]
        return indep_cols, dep_cols

    def get_entries_by_t(self):
        t_cols = {}
        t_eq_rows = {}
        for t in range(self.t):
            cols = [_ for _ in range(self.n_tot * t, self.n_tot * (t + 1))]
            cols += [self.n_tot * self.t + self.n_gen * t + _ for _ in range(self.n_gen)]
            t_cols[t] = cols

            tanks_rows = [self.t * self.n_bus + t * self.n_tanks + _ for _ in range(self.n_tanks)]
            rows = ([(t * self.n_bus) + _ for _ in range(self.n_bus)] + tanks_rows)
            t_eq_rows[t] = rows
        return t_cols, t_eq_rows

    def build_equality_constraints(self):
        """
        Construct the bus power balance matrix for the case of equality constraints
        variables order (matrix columns), for t in range(self.t):

        [θ0, θ1, ... θN | Xg0, Xg1, ... XgN | bat0C, Xbat1C, ... batNC | bat0D, Xbat1D, ... batND |
        comb0, comb1 ... combN | desal0, desal1 ... deslaN | tank0, tank1 ... tankN]

        In the end are the piecewise linear variables:
        [pw_g0_t0, pw_g1_t0 ... pw_G_t0, ... pw_g0_T, pw_g1_T ... pw_G_T]
        """
        # preparing batteries columns
        if self.n_bat_vars == 2:
            bat_cols = utils.get_mat_for_type(self.pds.bus, self.pds.batteries)
            bat_cols = bat_cols[:, np.any(bat_cols, axis=0)]
            charge_eff = self.pds.bus["charge_eff"].values.reshape(-1, 1)
            bat_c_cols = np.divide(bat_cols, charge_eff, out=np.full_like(bat_cols, 0, dtype=float),
                                   where=charge_eff != 0)
            bat_c_cols[np.abs(bat_c_cols) < 10 ** -6] = 0
            discharge_eff = self.pds.bus["discharge_eff"].values.reshape(-1, 1)
            bat_d_cols = np.multiply(bat_cols, discharge_eff)
            bat_d_cols[np.abs(bat_d_cols) < 10 ** -6] = 0
        else:
            bat_cols = utils.get_mat_for_type(self.pds.bus, self.pds.batteries)
            bat_c_cols = bat_cols[:, np.any(bat_cols, axis=0)]

        # preparing pds-wds links columns
        pumps_power, desal_power = self.construct_wds_pds_links()

        mat = np.zeros((self.n_bus * self.t + self.n_tanks * self.t, self.n_tot * self.t + self.n_pw_vars))
        for t in range(self.t):
            mat[self.n_bus * t: self.n_bus * (t + 1), self.n_tot * t: self.n_tot * t + self.n_bus] = self.pds.y

            gen_columns = utils.get_mat_for_type(self.pds.bus, self.pds.generators)
            gen_columns = gen_columns[:, np.any(gen_columns, axis=0)]
            mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + self.n_bus: self.n_tot * t + (self.n_bus + self.n_gen)] = gen_columns

            # batteries variables explicitly represent the net in/out from the battery
            # first block of columns for charging, second block for discharging
            if self.n_bat_vars == 2:
                mat[self.n_bus * t: self.n_bus * (t + 1),
                    self.n_tot * t + (self.n_bus + self.n_gen):
                    self.n_tot * t + (self.n_bus + self.n_gen) + self.n_bat] = -1 * bat_c_cols
                mat[self.n_bus * t: self.n_bus * (t + 1),
                    self.n_tot * t + (self.n_bus + self.n_gen + self.n_bat): self.n_tot * t + self.n_pds] = bat_d_cols
            else:
                mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + (self.n_bus + self.n_gen):
                self.n_tot * t + (self.n_bus + self.n_gen) + self.n_bat] = -1 * bat_c_cols

            # pumps combs variables inside power balance (multiplied by the combination nominal power)
            mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + self.n_pds: self.n_tot * t + (self.n_pds + self.n_combs)] = -1 * pumps_power

            # desalination variables inside power balance (with desalination power consumption)
            mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + (self.n_pds + self.n_combs):
                self.n_tot * t + (self.n_pds + self.n_combs) + self.n_desal] = -1 * desal_power

        # water balance
        start_row = self.n_bus * self.t
        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            for t in range(self.t):
                r = start_row + tank_idx + self.n_tanks * t
                for comb_idx, comb_data in self.wds.combs.iterrows():
                    col = self.n_tot * t + self.n_pds + comb_idx
                    if comb_data['to'] == tank_name:
                        mat[r, col] = comb_data['flow']
                    if comb_data['from'] == tank_name:
                        mat[r, col] = -comb_data['flow']

                for desal_idx, desal_data in self.wds.desal.iterrows():
                    if desal_data['to'] == tank_name:
                        col = self.n_tot * t + self.n_pds + self.n_combs + desal_idx
                        mat[r, col] = 1

                mat[r, self.n_tot * t + self.n_pds + self.n_combs + self.n_desal + tank_idx] = -1

        # constructing corresponding RHS
        loads, injections, demands = self.get_nominal_rhs()
        rhs = np.hstack([loads - injections, demands])  # add zeros for ref bust angles
        return mat, rhs

    def construct_wds_pds_links(self):
        if self.wds is None or self.wds.combs.empty:
            pumps_power = 0
        else:
            power_mat = self.wds.combs[[_ for _ in self.wds.combs.columns if _.startswith('power')]].fillna(0).values.T
            power_mat = power_mat * self.wds_power_units_factor
            pumps_power = self.pds.pumps_bus.values @ power_mat

        if self.wds is None or self.wds.desal.empty:
            desal_power = 0
        else:
            desal_power = (self.pds.desal_bus.values * self.wds.desal_power * self.wds_power_units_factor)

        return pumps_power, desal_power

    def get_nominal_rhs(self):
        loads = self.pds.dem_active.values[:, :self.t].flatten('F')
        injections = np.multiply(self.pds.bus['max_pv_pu'].values,
                                 self.pds.max_gen_profile.values[:, :self.t].T).T.flatten('F')

        demands = self.wds.demands.values[:self.t, :].flatten()
        return loads, injections, demands

    def get_x_bounds(self):
        bus_angles_lb = -np.tile(np.pi, self.n_bus)
        bus_angles_ub = np.tile(np.pi, self.n_bus)
        bus_angles_lb[0] = -0.000001
        bus_angles_ub[0] = 0.000001

        single_t_lb = np.hstack([bus_angles_lb,  # lower bound for bus angles
                                 self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'min_gen_p_pu'],  # generators lb
                                 np.tile(0, self.n_bat),  # lower bound for batteries charge
                                 np.tile(0, self.n_bat),  # lower bound for batteries discharge
                                 np.tile(0, self.n_combs),  # lower bound for combs duration
                                 self.wds.desal['min_flow'].values,  # desalination lower bound
                                 - np.tile(10 ** 4, self.n_tanks)  # tank inflows
                                 ])

        single_t_ub = np.hstack([bus_angles_ub,  # upper bound for bus angles
                                 self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'max_gen_p_pu'],
                                 self.pds.bus.loc[self.pds.bus['max_charge'] > 0, 'max_charge'],
                                 self.pds.bus.loc[self.pds.bus['max_discharge'] > 0, 'max_discharge'],
                                 np.tile(1, self.n_combs),  # upper bound for combs duration
                                 self.wds.desal['max_flow'].values,  # desalination upper bound
                                 np.tile(10 ** 4, self.n_tanks)  # tank inflows
                                 ])

        if self.n_bat_vars == 1:
            mask = np.ones(len(single_t_lb), dtype=bool)
            idx_to_drop = [self.n_bus + self.n_gen + _ + 1 for _ in range(self.n_bat)]
            mask[idx_to_drop] = False
            single_t_lb = single_t_lb[mask]
            single_t_ub = single_t_ub[mask]

            bat_discharge = -self.pds.bus.loc[self.pds.bus['max_discharge'] > 0, 'max_discharge']
            single_t_lb[self.n_bus + self.n_gen: self.n_pds] = bat_discharge

        lb = np.tile(single_t_lb, self.t)
        ub = np.tile(single_t_ub, self.t)

        if self.pw_segments is not None:
            pw_ub = 10 ** 5  # inf
            ub = np.hstack([ub, np.tile(pw_ub, self.n_pw_vars)])
            lb = np.hstack([lb, np.tile(0, self.n_pw_vars)])
        return lb, ub

    def build_inequality_constraints(self):
        self.piecewise_linear_costs()
        self.batteries_bounds()
        self.ramping()
        self.one_comb_only()
        self.water_mass_balance()

    def piecewise_linear_costs(self):
        v = self.w - 1  # back to number of segments
        mat = np.zeros((v * self.n_gen * self.t, self.n_tot * self.t + self.n_gen * self.t))
        rhs = np.ones(v * self.n_gen * self.t)

        for i, (gen_idx, row) in enumerate(self.pds.generators.iterrows()):
            bp = utils.quad_to_piecewise(a=row['gen_a'], b=row['gen_b'], c=row['gen_c'], p_min=row['min_gen_p'],
                                         p_max=row['max_gen_p'], num_segments=self.pw_segments)

            # convert power to pu, and scale costs for numeric stability
            bp = [(_[0] * self.pds.to_pu, _[1] / self.piecewise_coef_scale) for _ in bp]

            # y = a * x + b
            a = [((bp[_ - 1][1] - bp[_][1]) / (bp[_ - 1][0] - bp[_][0])) for _ in range(1, len(bp))]
            b = [-((bp[_ - 1][1] - bp[_][1]) / (bp[_ - 1][0] - bp[_][0])) * bp[_ - 1][0] + bp[_ - 1][1]
                 for _ in range(1, len(bp))]

            for t in range(self.t):
                mat[(self.n_gen * t * v) + i * v: (self.n_gen * t * v) + (i + 1) * v,
                    self.n_tot * t + self.n_bus + i] = a
                mat[(self.n_gen * t * v) + i * v: (self.n_gen * t * v) + (i + 1) * v,
                    self.n_tot * self.t + self.n_gen * t + i] = -1

                rhs[(self.n_gen * t * v) + i * v: (self.n_gen * t * v) + (i + 1) * v] = -np.array(b)
                self.piecewise_costs[self.n_tot * self.t + self.n_gen * t + i] = 1

        self.ineq_mat["pw"] = mat
        self.ineq_rhs["pw"] = rhs

    def batteries_bounds(self):
        bat_idx = self.pds.bus.loc[self.pds.bus['max_storage'] > 0].index  # filter by max to get bat
        bat_init = self.pds.bus.loc[bat_idx, 'init_storage'].values
        bat_final = self.pds.bus.loc[bat_idx, 'final_storage'].values
        bat_min = self.pds.bus.loc[bat_idx, 'min_storage'].values
        bat_max = self.pds.bus.loc[bat_idx, 'max_storage'].values

        # batteries balance - lb and ub for each battery for every time step
        mat = np.zeros((self.n_bat * 2 * self.t, self.n_tot * self.t + self.n_pw_vars))
        rhs = np.zeros((self.n_bat * 2 * self.t))
        for b in range(self.n_bat):
            for t in range(self.t):
                if self.n_bat_vars == 2:
                    charge_col = self.n_tot * t + (self.n_bus + self.n_gen) + b
                    discharge_col = self.n_tot * t + (self.n_bus + self.n_gen + self.n_bat) + b
                    start_row, end_row = 2 * b * self.t + t, (2 * b + 1) * self.t
                    # first t rows are used for lb for the 'b' battery
                    mat[start_row: end_row, charge_col] = -1
                    mat[start_row: end_row, discharge_col] = 1
                    # next block of t rows are used for ub for the 'b' battery
                    mat[start_row + self.t: end_row + self.t, charge_col] = 1
                    mat[start_row + self.t: end_row + self.t, discharge_col] = -1

                else:
                    charge_col = self.n_tot * t + (self.n_bus + self.n_gen) + b
                    start_row, end_row = 2 * b * self.t + t, (2 * b + 1) * self.t
                    # first t rows are used for lb for the 'b' battery
                    mat[start_row: end_row, charge_col] = -1
                    # next block of t rows are used for ub for the 'b' battery
                    mat[start_row + self.t: end_row + self.t, charge_col] = 1

            rhs[2 * b * self.t: (2 * b + 1) * self.t] = - bat_min[b] + bat_init[b]  # lb
            rhs[(2 * b + 1) * self.t - 1] = bat_init[b] - bat_final[b]  # final battery state of charge
            rhs[2 * b * self.t + self.t: (2 * b + 1) * self.t + self.t] = bat_max[b] - bat_init[b]  # ub

        self.ineq_mat["batteries_bounds"] = mat
        self.ineq_rhs["batteries_bounds"] = rhs

    def ramping(self):
        gen_buses = self.pds.bus.loc[self.pds.bus['type'] == "gen"]  # filter to gets generator idx
        mat = np.zeros((self.n_gen * self.t * 2, self.n_tot * self.t + self.n_pw_vars))
        rhs = np.ones(self.n_gen * self.t * 2)
        for i, (gen_idx, row) in enumerate(gen_buses.iterrows()):
            for t in range(self.t - 1):
                # first block for ramp up
                t_col = self.n_tot * t + self.n_bus + i
                t_plus_one_col = self.n_tot * (t + 1) + self.n_bus + i
                mat[2 * i * self.t + t, t_col] = -1
                mat[2 * i * self.t + t, t_plus_one_col] = 1
                # second block for ramp down
                mat[2 * i * self.t + t + self.t, t_col] = 1
                mat[2 * i * self.t + t + self.t, t_plus_one_col] = -1

            rhs[2 * i * self.t: (2 * i + 1) * self.t] = row['ramping']
            rhs[2 * i * self.t + self.t: (2 * i + 1) * self.t + self.t] = row['ramping']

        self.ineq_mat["ramping"] = mat
        self.ineq_rhs["ramping"] = rhs

    def one_comb_only(self):
        mat = np.zeros((self.wds.n_stations * self.t, self.n_tot * self.t + self.n_pw_vars))
        rhs = np.ones(self.wds.n_stations * self.t)
        c = 0
        for i, station in enumerate(self.wds.combs["station"].unique()):
            k = len(self.wds.combs.loc[self.wds.combs["station"] == station])  # num combs for station
            for t in range(self.t):
                mat[i * self.t + t, self.n_tot * t + self.n_pds + c: self.n_tot * t + self.n_pds + c + k] = 1
            c += k

        if c > 0:
            self.ineq_mat["one_comb"] = mat
            self.ineq_rhs["one_comb"] = rhs

    def water_mass_balance(self):
        mat = np.zeros((2 * self.n_tanks * self.t, self.n_tot * self.t + self.n_pw_vars))
        rhs = np.zeros(2 * self.n_tanks * self.t)
        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            for t in range(self.t):
                col = self.n_tot * t + self.n_pds + self.n_combs + self.n_desal + tank_idx
                start_row_lb = tank_idx * self.t + t
                end_row_lb = (tank_idx + 1) * self.t
                start_row_ub = self.n_tanks * self.t + tank_idx * self.t + t
                end_row_ub = self.n_tanks * self.t + (tank_idx + 1) * self.t

                mat[start_row_lb: end_row_lb, col] = -1  # lb: -Qin <= RHS
                mat[start_row_ub: end_row_ub, col] = 1  # ub: Qin <= RHS

            tank_start_row, tank_end_row = tank_idx * self.t, (tank_idx + 1) * self.t
            rhs[tank_start_row: tank_end_row] = tank_data['init_vol'] - tank_data['min_vol']  # lb
            rhs[tank_end_row - 1] = 0  # final tank vol

            tank_start_row = self.n_tanks * self.t + tank_idx * self.t  # for ub
            tank_end_row = self.n_tanks * self.t + (tank_idx + 1) * self.t  # for ub
            rhs[tank_start_row: tank_end_row] = - tank_data['init_vol'] + tank_data['max_vol']  # ub

        self.ineq_mat["mass_balance"] = mat
        self.ineq_rhs["mass_balance"] = rhs

    def get_variables_to_eliminate(self, mat, method='qr'):
        if method == 'rref':
            echelon, dep_idx = sp.Matrix(mat).rref()
            indep_idx = np.setxor1d(np.arange(mat.shape[1]), dep_idx)
            indep_idx = list(indep_idx)

        elif method == 'qr':
            q, r, p = scipy.linalg.qr(mat, pivoting=True)
            tol = 1e-10  # tolerance for detecting non-zero entries
            rank = np.sum(np.abs(np.diag(r)) > tol)
            independent_columns_indices = p[:rank]
            dep_idx = independent_columns_indices
            indep_idx = np.setxor1d(np.arange(mat.shape[1]), dep_idx)

            Q, R, P = scipy.linalg.qr(mat, pivoting=True)
            diag = np.abs(np.diag(R))
            rank = np.sum(diag > tol)
            dep_idx = sorted(P[:rank].tolist())
            indep_idx = sorted(P[rank:].tolist())

        elif method == 'svd':
            u, sigma, vt = np.linalg.svd(mat, full_matrices=False)
            tol = max(mat.shape) * np.max(sigma) * np.finfo(float).eps
            rank = np.sum(sigma > tol)
            dep_idx = np.arange(mat.shape[1])[:rank]
            indep_idx = np.arange(mat.shape[1])[rank:]

        elif method == 'manual':
            indep_idx, dep_idx = self.manually_decompose_variables()

        dep_idx, indep_idx = sorted(list(dep_idx)), sorted(list(indep_idx))
        return dep_idx, indep_idx

    def manually_decompose_variables(self):
        """
        Build the independent and dependent variables from an input list by the user
        The list contain the independent variables of the first time step (not including piecewise linear)
        Based on this list the independent variables are duplicated to include all time steps
        and the piecewise linear variables which are always independent are added
        :return:
        """
        piecewise_linear_vars = [self.n_tot * self.t + _ for _ in range(self.n_gen)]  # always indep
        indep_0 = self.manual_indep_variables + piecewise_linear_vars
        dep_0 = [_ for _ in range(self.n_tot) if _ not in indep_0]
        indep_all, dep_all = [], []

        for i in indep_0:
            if i <= self.n_tot:
                indep_all += [i + self.n_tot * _ for _ in range(self.t)]
            else:
                indep_all += [i + self.n_gen * _ for _ in range(self.t)]  # pw variables

        for i in dep_0:
            if i <= self.n_tot:
                dep_all += [i + self.n_tot * _ for _ in range(self.t)]
            else:
                dep_all += [i + self.n_gen * _ for _ in range(self.t)]  # pw variables

        indep_all = sorted(indep_all)
        dep_all = sorted(dep_all)
        return indep_all, dep_all

    def decompose_x(self, x):
        theta = np.zeros((self.n_bus, self.t))
        gen = np.zeros((self.n_gen, self.t))
        bat_c = np.zeros((self.n_bat, self.t))
        bat_d = np.zeros((self.n_bat, self.t))
        pumps = np.zeros((self.n_combs, self.t))
        desal = np.zeros((self.n_desal, self.t))
        tanks = np.zeros((self.n_tanks, self.t))
        w = np.zeros((self.n_pw_vars // self.t, self.t))
        for t in range(self.t):
            theta[:, t] = x[self.n_tot * t: self.n_tot * t + self.n_bus]
            gen[:, t] = x[self.n_tot * t + self.n_bus: self.n_tot * t + self.n_bus + self.n_gen]
            bat_c[:, t] = x[self.n_tot * t + self.n_bus + self.n_gen: self.n_tot * t + self.n_pds - self.n_bat]
            bat_d[:, t] = x[self.n_tot * t + self.n_bus + self.n_gen + self.n_bat: self.n_tot * t + self.n_pds]
            pumps[:, t] = x[self.n_tot * t + self.n_pds: self.n_tot * t + self.n_pds + self.n_combs]
            desal[:, t] = x[self.n_tot * t + self.n_pds + self.n_combs:
                            self.n_tot * t + self.n_pds + self.n_combs + self.n_desal]
            tanks[:, t] = x[self.n_tot * t + self.n_pds + self.n_combs + self.n_desal:
                            self.n_tot * t + self.n_pds + self.n_wds]
            # w[:, t] = x[self.n_tot * self.t + self.n_gen * t]

        return {'theta': theta, 'gen_p': gen, 'bat_c': bat_c, 'bat_d': bat_d, 'w': w,
                'pumps': pumps, 'desal': desal, 'tanks': tanks}


class StandardDCPF(BaseOptModel):
    """
    Based on standard equality constraints
    """

    def __init__(self, pds_path, wds_path, t, omega=None, opt_method=None, elimination_method=None,
                 manual_indep_variables=None, pw_segments=None, n_bat_vars=2, solver_params=None, solver_display=False,
                 **kwargs):
        super().__init__(pds_path, wds_path, t, omega, opt_method, elimination_method, manual_indep_variables,
                         pw_segments, n_bat_vars, solver_params, solver_display, **kwargs)

        self.lb, self.ub = self.get_x_bounds()
        self.x = self.declare_variables()
        self.formulate()

    def declare_variables(self):
        x = cp.Variable(self.eq_mat.shape[1])
        self.constraints += [x <= self.ub, x >= self.lb]
        return x

    def formulate(self):
        self.constraints.append(self.eq_mat @ self.x == self.eq_rhs)
        for (name_mat, mat), (name_rhs, rhs) in zip(self.ineq_mat.items(), self.ineq_rhs.items()):
            self.constraints.append(mat @ self.x <= rhs)

    def solve(self):
        self.problem = cp.Problem(cp.Minimize(self.piecewise_costs @ self.x), self.constraints)
        self.problem.solve(solver=cp.GUROBI, reoptimize=True)
        obj, status, solver_time = self.problem.value, self.problem.status, self.problem.solver_stats.solve_time

        run_time = self.problem.solver_stats.solve_time
        print(f"Objective (WC): {self.problem.value} | Solver time: {run_time}")
        return obj, status, solver_time


class RobustModel(BaseOptModel):
    """
    Robust and Adjustable Robust Optimization
    """
    def __init__(self, pds_path, wds_path, t, omega, opt_method, elimination_method, manual_indep_variables=None,
                 pw_segments=None, n_bat_vars=2, solver_params=None, solver_display=False,
                 pds_lat=0, wds_lat=0, pds_lags=None, wds_lags=None, reduced_bound_constraints=True,
                 plot=False, **kwargs):
        super().__init__(pds_path, wds_path, t, omega, opt_method, elimination_method, manual_indep_variables,
                         pw_segments, n_bat_vars, solver_params, solver_display, **kwargs)

        self.pds_lat = pds_lat
        self.wds_lat = wds_lat
        self.pds_lags = pds_lags
        self.wds_lags = wds_lags
        if self.pds_lags is None:
            self.pds_lags = self.t

        if self.wds_lags is None:
            self.wds_lags = self.t

        self.reduced_bound_constraints = reduced_bound_constraints
        self.plot = plot

        # matrices for time based formulation
        self.p1 = None
        self.p2 = None
        self.k1 = None
        self.k2 = None
        self.A1 = None
        self.A2 = None
        # full matrices for vectorized formulation
        self._A1 = None
        self._A2 = None
        self._p1 = None
        self._p2 = None
        self._k1 = None
        self._k2 = None
        self._k = None

        self.t_formulation_start = None
        self.z0 = None
        self.z1 = None
        self.z0_val = None
        self.z1_val = None

        self.n_uncertain_per_t = None
        self.n_uncertain_bus_per_t = None
        self.z_to_b_map = None
        self.ldr_to_z_map = {}

        self.obj = None
        self.x = None
        self.x_by_sample = None
        self.n_indep_per_t = None
        self.n_dep_per_t = None
        self.dep_idx_t, self.indep_idx_t = {}, {}

        # Ax <= b
        self.A = copy.deepcopy(self.eq_mat)
        self.b = copy.deepcopy(self.eq_rhs)
        # Bx <= c
        self.B = np.vstack([_ for _ in self.ineq_mat.values()])
        self.c = np.hstack([_ for _ in self.ineq_rhs.values()])

        if self.reduced_bound_constraints:
            var_bounds_mat, var_bounds_rhs = self.build_reduced_range_constraints()
            self.B = np.vstack([self.B, var_bounds_mat])
            self.c = np.hstack([self.c, var_bounds_rhs])
        else:
            self.lb, self.ub = self.get_x_bounds()
            self.B = np.vstack([self.B, -np.eye(self.B.shape[1]), np.eye(self.B.shape[1])])
            self.c = np.hstack([self.c, -self.lb, self.ub])

        self.B = sparse.csr_matrix(self.B)

        self.dep_idx, self.indep_idx = self.get_variables_to_eliminate(mat=self.A, method=self.elimination_method)
        self.cov, self.delta = uncertainty.affine_mat(self.pds, self.wds, self.t)
        r = self.project_u_to_rhs()  # r is the projection matrix
        projected_cov = r @ self.cov @ r.T
        self.projected_delta = uncertainty.cholesky_with_zeros(projected_cov)
        self.do_math()

    def build_reduced_range_constraints(self):
        """
        A reduced version of the range constraints - Not all variables must have LB and UB
        The ranges defined in BaseOptModel assign LB and UB for *all* variables, although some of them are redundant
        To avoid numerical issues (scaling) and decrease memory usage, here we only assign bounds where needed

        Memory is critical.
        The constraints are second-order-cones. The solver will use epigraph form and add auxiliary
        variables for each constraint. Therefore, it is essential to reduce the constraints number as much as possible

        Example for redundant bounds:
            tanks inflow not have to be bounded
            voltage angles (beside reference bus)
        """
        ref_bus_angles_lb = -0.000000001
        ref_bus_angles_ub = 0.000000001

        single_t_lb = np.hstack([ref_bus_angles_lb,  # lower bound for ref bus angle
                                 self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'min_gen_p_pu'],  # generators lb
                                 np.tile(0, self.n_bat),  # lower bound for batteries charge
                                 np.tile(0, self.n_bat),  # lower bound for batteries discharge
                                 np.tile(0, self.n_combs),  # lower bound for combs duration
                                 self.wds.desal['min_flow'].values,  # desalination lower bound
                                 ])

        single_t_ub = np.hstack([ref_bus_angles_ub,  # upper bound for ref bus angle
                                 self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'max_gen_p_pu'],
                                 self.pds.bus.loc[self.pds.bus['max_charge'] > 0, 'max_charge'],
                                 self.pds.bus.loc[self.pds.bus['max_discharge'] > 0, 'max_discharge'],
                                 np.tile(1, self.n_combs),  # upper bound for combs duration
                                 self.wds.desal['max_flow'].values,  # desalination upper bound
                                 ])

        if self.n_bat_vars == 1:
            mask = np.ones(len(single_t_lb), dtype=bool)
            idx_to_drop = [1 + self.n_gen + _ + 1 for _ in range(self.n_bat)]
            mask[idx_to_drop] = False
            single_t_lb = single_t_lb[mask]
            single_t_ub = single_t_ub[mask]

            bat_discharge = -self.pds.bus.loc[self.pds.bus['max_discharge'] > 0, 'max_discharge']
            single_t_lb[1 + self.n_gen: self.n_pds] = bat_discharge

        lb = np.tile(single_t_lb, self.t)
        ub = np.tile(single_t_ub, self.t)

        if self.pw_segments is not None:
            lb = np.hstack([lb, np.tile(0, self.n_pw_vars)])

        a1 = np.eye(self.n_tot)  # if we add lb, ub for all variables
        a1[:, 1: self.n_bus] = 0  # remove bounds for all bus angles except ref bus
        a1[:, self.n_pds + self.n_combs + self.n_desal: self.n_pds + self.n_wds] = 0  # remove bounds for tanks inflows

        a1 = np.kron(np.eye(self.t, dtype=a1.dtype), a1)
        a1 = a1[~np.all(a1 == 0, axis=1)]  # drop the rows where constraints are not needed
        a = np.block([[-a1, np.zeros((a1.shape[0], self.n_pw_vars))],
                      [np.zeros((self.n_pw_vars, a1.shape[1])), -np.eye(self.n_pw_vars)],
                      [a1, np.zeros((a1.shape[0], self.n_pw_vars))]])

        b = np.hstack([-lb, ub])
        pd.DataFrame(np.hstack([a, b.T.reshape(-1, 1)])).to_csv("a.csv")
        return a, b

    def export_matrix(self):
        pd.DataFrame(np.hstack([self.A, self.b.reshape(-1, 1)])).to_csv("eq_mat.csv")
        pd.DataFrame(np.hstack([self.B.toarray(), self.c.reshape(-1, 1)])).to_csv("ineq_mat.csv")

    def do_math(self):
        for t in range(self.t):
            self.dep_idx_t[t] = sorted(list(set(self.dep_idx) & set(self.t_cols[t])))
            self.indep_idx_t[t] = sorted(list(set(self.indep_idx) & set(self.t_cols[t])))

        self.n_indep_per_t = len(self.indep_idx_t[0])
        self.n_dep_per_t = len(self.dep_idx_t[0])
        self.A1 = {t: self.A[np.ix_(self.t_eq_rows[t], self.dep_idx_t[t])] for t in range(self.t)}
        self.A2 = {t: self.A[np.ix_(self.t_eq_rows[t], self.indep_idx_t[t])] for t in range(self.t)}

        n_vars_per_t = len(self.t_cols[0])
        dep_relative_idx, indep_relative_idx = utils.get_relative_indices(self.dep_idx_t[0], self.indep_idx_t[0])
        self.p1 = np.zeros((n_vars_per_t, len(self.indep_idx_t[0])))
        self.p1[indep_relative_idx, :] = np.eye(len(self.indep_idx_t[0]))
        self.p2 = np.zeros((n_vars_per_t, len(self.dep_idx_t[0])))
        self.p2[dep_relative_idx, :] = np.eye(len(self.dep_idx_t[0]))
        self.k1 = {t: sparse.csr_matrix(self.p2 @ np.linalg.pinv(self.A1[t])) for t in range(self.t)}
        self.k2 = {t: sparse.csr_matrix(self.p1 - (self.p2 @ np.linalg.pinv(self.A1[t])) @ self.A2[t]) for t in range(self.t)}

        certain_bus_per_t = [_ for _ in self.get_certain_bus() if _ < self.n_bus]
        n_certain_bus_per_t = len(certain_bus_per_t)
        self.n_uncertain_per_t = self.n_bus - n_certain_bus_per_t + self.n_tanks
        self.n_uncertain_bus_per_t = self.n_bus - n_certain_bus_per_t
        uncertain_bus = [_ for _ in range(self.n_bus * self.t) if _ not in self.get_certain_bus()]
        if self.opt_method == "ro":
            self.z_to_b_map = np.zeros((self.n_uncertain_per_t * self.t, self.b.shape[0]))
        else:
            self.z_to_b_map = np.zeros((self.n_uncertain_per_t * self.t, self.b.shape[0]))
            self.z_to_b_map[:len(uncertain_bus), uncertain_bus] = np.eye(len(uncertain_bus))
            self.z_to_b_map[-self.n_tanks * self.t:, -self.n_tanks * self.t:] = np.eye(self.n_tanks * self.t)
            self.z_to_b_map = sparse.csr_matrix(self.z_to_b_map)

        # complete matrices
        self._A1 = self.A[:, self.dep_idx]
        self._A2 = self.A[:, self.indep_idx]
        self._p1 = np.zeros((self.A.shape[1], len(self.indep_idx)))
        self._p1[self.indep_idx, :] = np.eye(len(self.indep_idx))
        self._p2 = np.zeros((self.A.shape[1], len(self.dep_idx)))
        self._p2[self.dep_idx, :] = np.eye(len(self.dep_idx))
        self._k1 = sparse.csr_matrix(self._p2 @ np.linalg.pinv(self._A1))
        self._k2 = sparse.csr_matrix(self._p1 - (self._p2 @ np.linalg.pinv(self._A1)) @ self._A2)
        # self._k = self._p2 @ np.linalg.pinv(self._A1) @ self._A2
        self._k = self._p1 - (self._p2 @ np.linalg.pinv(self._A1) @ self._A2)

    @staticmethod
    def get_ldr_block(n, k, T, lags, lat):
        total_size = (n * T, k * T)
        mat = np.zeros(total_size)
        for i in range(lags):
            # Start filling from (lat + i) to respect the lat offset
            for j in range(lat + i, T):
                if (j - i) >= 1:  # Ensure that we are placing on correct diagonals below main
                    start_row = j * n
                    end_row = start_row + n
                    start_col = (j - i - 1 - lat) * k
                    end_col = start_col + k
                    mat[start_row:end_row, start_col:end_col] = 1
        return mat

    def build_nonanticipative_matrix(self, n_lags=None, lat=0):
        if n_lags is None:
            n_lags = self.t

        certain_bus_per_t = [_ for _ in self.get_certain_bus() if _ < self.n_bus]
        n_certain_bus_per_t = len(certain_bus_per_t)
        loads_block = self.get_ldr_block(n=self.n_indep_per_t, k=self.n_bus - n_certain_bus_per_t, T=self.t,
                                         lags=n_lags, lat=self.pds_lat)
        dem_block = self.get_ldr_block(n=self.n_indep_per_t, k=self.n_tanks, T=self.t, lags=n_lags, lat=self.wds_lat)
        mat = np.block([loads_block, dem_block])
        return mat

    def get_certain_bus(self):
        """
        get the indexes of buses with no loads and no pv - rhs should be 0 with no uncertainty
        """
        certain_load_idx = np.where(self.pds.dem_active_std.iloc[:, :self.t].values.flatten('F') == 0)[0]
        certain_pv_idx = np.where(self.pds.pv_std[:, :self.t].flatten('F') == 0)[0]
        certain_idx = list(set(certain_load_idx) & set(certain_pv_idx))
        return certain_idx

    def project_u_to_rhs(self):
        """
        from uncertainty set dimension to equality rhs dimension
        uncertainty set dimension is: [loads (n_bus) + pv (n_bus) + dem (n_tanks)] * T
        equality rhs dimension is: [power balance (n_bus) ref bus (1) + dem (n_tanks)] * T
        """
        mat = np.zeros(((self.n_bus + self.n_tanks) * self.t, (self.n_bus + self.n_bus + self.n_tanks) * self.t))

        # power balance rows
        for t in range(self.t):
            e = np.eye(self.t)[t]
            for b in range(self.n_bus):
                mat[self.n_bus * t + b, b * self.t: (b + 1) * self.t] = e
                mat[self.n_bus * t + b, self.n_bus * self.t + b * self.t: self.n_bus * self.t + (b + 1) * self.t] = -e

        # water demand rows
        row0 = self.n_bus * self.t  # after power balance rows
        col0 = self.n_bus * self.t + self.n_bus * self.t  # after cols for loads and pv uncertainty
        for t in range(self.t):
            e = np.eye(self.t)[t]
            for tank_idx in range(self.n_tanks):
                mat[row0 + self.n_tanks * t + tank_idx, col0 + tank_idx * self.t: col0 + (tank_idx + 1) * self.t] = e
        return mat

    def formulate_ro(self):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()
        self.z0 = cp.Variable(self.n_indep_per_t * self.t)
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)

        w0 = sum(
            (self.B[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
            for t in range(self.t))
        w1 = sum((self.B[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                 for t in range(self.t))

        for j in range(self.B.shape[0]):
            self.constraints.append(-self.c[j] + w0[j] + (w1 @ self.b)[j]
                                    + self.omega_param * cp.norm((w1 @ self.projected_delta)[j], 2)
                                    <= 0)

        # formulate the objective function as constraint
        f = self.piecewise_costs.reshape(1, -1)
        Obj0 = sum((f[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
                   for t in range(self.t))
        Obj1 = sum((f[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                   for t in range(self.t))
        self.constraints.append(-self.obj + Obj0 + Obj1 @ self.b
                                + self.omega_param * cp.norm(Obj1 @ self.projected_delta, 2)
                                <= 0)

        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def formulate_vectorized_opt_problem(self):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()

        n_indep_no_pw = self.n_indep_per_t - self.n_gen
        certain_bus_per_t = [_ for _ in self.get_certain_bus() if _ < self.n_bus]
        n_uncertain_bus_per_t = self.n_bus - len(certain_bus_per_t)
        loads_block = self.get_ldr_block(n=n_indep_no_pw, k=n_uncertain_bus_per_t, T=self.t,
                                         lags=self.pds_lags, lat=self.pds_lat)
        dem_block = self.get_ldr_block(n=n_indep_no_pw, k=self.n_tanks, T=self.t, lags=self.wds_lags, lat=self.wds_lat)
        pw_loads_block = self.get_ldr_block(n=self.n_gen, k=n_uncertain_bus_per_t, T=self.t,
                                            lags=self.pds_lags, lat=self.pds_lat)
        pw_dem_block = self.get_ldr_block(n=self.n_gen, k=self.n_tanks, T=self.t, lags=self.wds_lags, lat=self.wds_lat)
        nonanticipative_mat = np.block([[loads_block, dem_block], [pw_loads_block, pw_dem_block]])

        # flip nonanticipative mat - constraint is on the elements not included
        nonanticipative_mat = 1 - nonanticipative_mat
        self.z0 = cp.Variable(self.n_indep_per_t * self.t)
        self.z1 = cp.Variable((self.n_indep_per_t * self.t, self.n_uncertain_per_t * self.t),
                              sparsity=np.where(nonanticipative_mat == 0))
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)

        B_k2 = self.B.dot(self._k2)
        B_k1 = self.B.dot(self._k1)

        B_k2 = B_k2.astype('float32')
        B_k1 = B_k1.astype('float32')
        self.z_to_b_map = self.z_to_b_map.astype('float32')
        self.c = self.c.astype('float32')
        self.b = self.b.astype('float32')
        self.projected_delta = sparse.csr_matrix(self.projected_delta.astype("float32"))
        w0 = B_k2 @ self.z0
        w1 = B_k1 + B_k2 @ self.z1 @ self.z_to_b_map

        # modeling 1 - fully vectorized
        constraint_vector = -self.c + w0 + w1 @ self.b + self.omega_param * cp.norm(w1 @ self.projected_delta, axis=1)
        self.constraints.append(constraint_vector <= 0)

        if self.opt_method == 'ro' or self.opt_method == 'det':
            self.constraints.append(self.z1 == 0)

        # modeling 2 - SOC constraint
        #  can be formulaized as: omega * cp.norm(w1 @ delta, axis=1) <= c - w0 - w1 @ b
        # w1_b = w1 @ self.b
        # t = cp.Variable(w1.shape[0])
        # for j in range(w1.shape[0]):
        #     self.constraints.append(cp.SOC(t[j], (w1 @ self.projected_delta)[j, :]))
        # self.constraints.append(self.omega_param * t <= self.c - w0 - w1_b)

        # modeling 3 - Epigraphs norm constraint
        # w1_b = w1 @ self.b
        # t = cp.Variable(w1.shape[0])
        # self.constraints.append(cp.norm(w1 @ self.projected_delta, axis=1) <= t)
        # self.constraints.append(self.omega_param * t <= self.c - w0 - w1_b)

        # modeling 4 - split constraints to batches of rows - control the tradeoff between memory usage and run times
        # batch_size = 100
        # w1_b = w1 @ self.b
        # for i in range(0, self.B.shape[0], batch_size):
        #     batch_indices = slice(i, min(i + batch_size, self.B.shape[0]))
        #
        #     # Compute the norm for the current batch.
        #     # Instead of computing norm over all rows at once, we slice out only the current rows.
        #     # (w1[batch_indices, :] @ self.projected_delta) has shape (batch_size, k)
        #     norm_batch = cp.norm(w1[batch_indices, :] @ self.projected_delta, axis=1)
        #
        #     # Build the constraint for this batch:
        #     c = -self.c[batch_indices] + w0[batch_indices] + w1_b[batch_indices] + self.omega_param * norm_batch <= 0
        #     self.constraints.append(c)

        f = self.piecewise_costs.reshape(1, -1)
        f_k2 = f.dot(self._k2.toarray())
        f_k1 = f.dot(self._k1.toarray())
        f_k2 = sparse.csr_matrix(f_k2.astype('float32'))
        f_k1 = sparse.csr_matrix(f_k1.astype('float32'))
        Obj0 = f_k2 @ self.z0
        Obj1 = f_k1 + f_k2 @ self.z1 @ self.z_to_b_map
        self.constraints.append(
            -self.obj + Obj0 + Obj1 @ self.b + self.omega_param * cp.norm(Obj1 @ self.projected_delta, axis=1) <= 0)
        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def formulate_opt_problem(self):
        """
        Not in use: old version where formulation is done by summing time periods components
        """
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()

        nonanticipative_mat = self.build_nonanticipative_matrix()
        # flip nonanticipative mat - constraint is on the elements not included
        nonanticipative_mat = 1 - nonanticipative_mat

        self.z0 = cp.Variable(self.n_indep_per_t * self.t)
        self.z1 = cp.Variable((self.n_indep_per_t * self.t, self.n_uncertain_per_t * self.t),
                              sparsity=np.where(nonanticipative_mat == 0))
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)

        w0 = sum(
            (self.B[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
            for t in range(self.t))
        w1 = sum((self.B[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                 + (self.B[:, self.t_cols[t]] @ self.k2[t])
                 @ self.z1[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map
                 for t in range(self.t))

        for j in range(self.B.shape[0]):
            self.constraints.append(-self.c[j] + w0[j] + (w1 @ self.b)[j]
                                    + self.omega_param * cp.norm((w1 @ self.projected_delta)[j], 2)
                                    <= 0)

        # formulate the objective function as constraint
        f = self.piecewise_costs.reshape(1, -1)
        Obj0 = sum((f[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
                   for t in range(self.t))
        Obj1 = sum((f[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                   + (f[:, self.t_cols[t]] @ self.k2[t])
                   @ self.z1[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map
                   for t in range(self.t))
        self.constraints.append(-self.obj + Obj0 + Obj1 @ self.b
                                + self.omega_param * cp.norm(Obj1 @ self.projected_delta, 2) <= 0)

        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def formulate_reduced_opt_problem(self):
        """
        not in use - old version
        """
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()
        self.z0 = cp.Variable(self.n_indep_per_t * self.t)
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)

        self.z1 = {}
        self.ldr_to_z_map = {}
        for t in range(self.t):
            if t > 0:
                self.z1[t] = cp.Variable((self.n_indep_per_t, self.n_uncertain_per_t * t))
                self.ldr_to_z_map[t] = np.zeros((self.n_uncertain_per_t * t, self.n_uncertain_per_t * self.t))
                uncertain_bus = [_ for _ in range(self.n_bus * t) if _ not in self.get_certain_bus()]
                self.ldr_to_z_map[t][:len(uncertain_bus), :len(uncertain_bus)] = np.eye(len(uncertain_bus))
                self.ldr_to_z_map[t][-self.n_tanks * t:,
                                    self.n_uncertain_bus_per_t * self.t:
                                    self.n_uncertain_bus_per_t * self.t + self.n_tanks * t] = np.eye(self.n_tanks * t)
            else:
                self.z1[t] = np.zeros((self.k2[t].shape[1], self.n_uncertain_per_t * self.t))
                self.ldr_to_z_map[t] = np.zeros((self.n_uncertain_per_t * self.t, self.n_uncertain_per_t * self.t))

        """
        to do - update this function to allow use of pds_lat and wds_lat
        the below code need to be tested before usage
        """
        # certain_bus_per_t = [_ for _ in self.get_certain_bus() if _ < self.n_bus]
        # n_certain_bus_per_t = len(certain_bus_per_t)
        # n_uncertain_bus_per_t = self.n_bus - n_certain_bus_per_t
        # for t in range(self.t):
        #     pds_ldr_dim = n_uncertain_bus_per_t * max(0, t - self.pds_lat)
        #     wds_ldr_dim = self.n_tanks * max(0, t - self.wds_lat)
        #     ldr_dim = pds_ldr_dim + wds_ldr_dim
        #     if ldr_dim > 0:
        #         self.z1[t] = cp.Variable((self.n_indep_per_t, ldr_dim))
        #         self.ldr_to_z_map[t] = np.zeros((ldr_dim, self.n_uncertain_per_t * self.t))
        #         uncertain_bus = [_ for _ in range(self.n_bus * (t - self.pds_lat)) if _ not in self.get_certain_bus()]
        #         self.ldr_to_z_map[t][:len(uncertain_bus), :len(uncertain_bus)] = np.eye(len(uncertain_bus))
        #         self.ldr_to_z_map[t][-self.n_tanks * (t - self.wds_lat):,
        #                             self.n_uncertain_bus_per_t * self.t:
        #                             self.n_uncertain_bus_per_t * self.t + self.n_tanks * (t - self.wds_lat)]\
        #             = np.eye(self.n_tanks * (t - self.wds_lat))
        #     else:
        #         self.z1[t] = np.zeros((self.k2[t].shape[1], self.n_uncertain_per_t * self.t))
        #         self.ldr_to_z_map[t] = np.zeros((self.n_uncertain_per_t * self.t, self.n_uncertain_per_t * self.t))

        w0 = sum(
            (self.B[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
            for t in range(self.t))
        w1 = sum((self.B[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                 + (self.B[:, self.t_cols[t]] @ self.k2[t]) @ self.z1[t] @ self.ldr_to_z_map[t] @ self.z_to_b_map
                 for t in range(self.t))

        for j in range(self.B.shape[0]):
            self.constraints.append(-self.c[j] + w0[j] + (w1 @ self.b)[j]
                                    + self.omega_param * cp.norm((w1 @ self.projected_delta)[j], 2) <= 0)

        # formulate the objective function as constraint
        f = self.piecewise_costs.reshape(1, -1)
        Obj0 = sum((f[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
                   for t in range(self.t))
        Obj1 = sum((f[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                   + (f[:, self.t_cols[t]] @ self.k2[t]) @ self.z1[t] @ self.ldr_to_z_map[t] @ self.z_to_b_map
                   for t in range(self.t))
        self.constraints.append(-self.obj + Obj0 + Obj1 @ self.b
                                + self.omega_param * cp.norm(Obj1 @ self.projected_delta, 2) <= 0)

        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def solve(self):
        self.omega_param.value = self.omega
        formulation_time = time.time() - self.t_formulation_start
        self.problem.solve(
            solver=cp.GUROBI,
            verbose=0,
            reoptimize=True,
            # canon_backend=cp.SCIPY_CANON_BACKEND,
            # ignore_dpp=True,
            BarHomogeneous=1, NumericFocus=1,  # for numeric stability
            Threads=10,
            BarConvTol=1e-3, FeasibilityTol=1e-4,  # for run time
            OutputFlag=0
                           )
        run_time = self.problem.solver_stats.solve_time
        print(f"Objective (WC): {self.problem.value * self.piecewise_coef_scale} | Formulation time: {formulation_time:.2f} "
              f"| Solver time: {run_time}")

        self.z0_val = self.z0.value
        if self.z1 is None:
            self.z1_val = np.zeros((self.n_indep_per_t * self.t, self.n_uncertain_per_t * self.t))
        elif isinstance(self.z1, cp.Variable):
            self.z1_val = self.z1.value
        elif isinstance(self.z1, dict):
            # this is for the reduced problem formulation - z1 is a dict of cvxpy variables
            self.z1_val = np.zeros((self.n_indep_per_t * self.t, self.n_uncertain_per_t * self.t))
            for t in range(self.t):
                if t > 0:
                    self.z1_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :]\
                        = self.z1[t].value @ self.ldr_to_z_map[t]
        else:
            self.z1_val = self.z1.value
        self.extract_solution()
        self.record_solution_metrics()

    def record_solution_metrics(self):
        self.solution_metrics['solver_name'] = self.problem.solver_stats.solver_name
        self.solution_metrics['status'] = self.problem.status
        self.solution_metrics['solver_time'] = self.problem.solver_stats.solve_time  # Time reported by the solver

        # The number of seconds it took to compile the problem the last time it was compiled
        self.solution_metrics['compilation_time'] = self.problem.compilation_time

        # The time (in seconds) it took for the solver to setup the problem.
        # self.solution_metrics['setup_time'] = self.problem.solver_stats.setup_time
        # self.solution_metrics['num_iterations'] = self.problem.solver_stats.num_iters

        self.solution_metrics['num_variables'] = self.problem.size_metrics.num_scalar_variables
        self.solution_metrics['num_constraints'] = self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr

        self.solution_metrics['high_level_variables'] = int(self.z0.size) + int(self.z1_sparsity_mask.sum()) + 1  # +1 for obj

        constraint_types = {}
        constraint_sizes = {}
        constraint_details = []

        for con in self.problem.constraints:
            con_type = type(con).__name__

            # Count constraint types
            if con_type not in constraint_types:
                constraint_types[con_type] = 0
            constraint_types[con_type] += 1

            # Track constraint sizes
            if con_type not in constraint_sizes:
                constraint_sizes[con_type] = []
            constraint_sizes[con_type].append(con.size)

            # Get more detailed information about each constraint
            constraint_info = {
                'type': con_type,
                'size': con.size,
                'expression': str(type(con.expr).__name__) if hasattr(con, 'expr') else None,
            }

            is_soc = False
            if con_type in ['SOC', 'SecondOrderCone']:
                is_soc = True
            elif hasattr(con, 'expr'):
                expr = con.expr
                # Check if expression involves norms or SOC-related atoms
                expr_str = str(type(expr).__name__)
                if any(keyword in expr_str.lower() for keyword in ['norm', 'soc', 'quadover', 'pnorm']):
                    is_soc = True
                # Also check the actual string representation
                expr_repr = str(expr)
                if 'norm2' in expr_repr.lower() or 'norm(' in expr_repr.lower():
                    is_soc = True
                # Check the arguments recursively
                if hasattr(expr, 'args'):
                    for arg in expr.args:
                        arg_type = str(type(arg).__name__)
                        if any(keyword in arg_type.lower() for keyword in ['norm', 'soc', 'quadover', 'quad', 'pnorm']):
                            is_soc = True
                            break
                        # Also check string representation of args
                        if 'norm2' in str(arg).lower():
                            is_soc = True
                            break

            constraint_info['is_soc'] = is_soc
            constraint_details.append(constraint_info)

        self.solution_metrics['constraints_by_type'] = constraint_types
        # self.solution_metrics['constraint_sizes_by_type'] = {
        #     k: {'count': len(v), 'total_size': sum(v), 'sizes': v}
        #     for k, v in constraint_sizes.items()
        # }
        # self.solution_metrics['constraint_details'] = constraint_details

        # Count SOC constraints specifically
        soc_count = sum([c['size'] for c in constraint_details if c['is_soc']])
        self.solution_metrics['num_soc_constraints'] = soc_count

        # sparsity
        solvers_to_try = [cp.GUROBI, cp.MOSEK, cp.SCS, cp.ECOS, cp.CLARABEL]
        problem_data = None
        solver_used = None

        for solver in solvers_to_try:
            try:
                problem_data = self.problem.get_problem_data(solver)
                solver_used = solver
                break
            except:
                continue

        if problem_data is None:
            print("Warning: Could not compile problem with any available solver")
            return

        data = problem_data[0]

        # Get constraint matrix - try multiple possible keys
        constraint_matrix = None
        matrix_keys = ['A', 'G', 'F', 'c']  # Different solvers use different keys

        for key in matrix_keys:
            if key in data and data[key] is not None:
                mat = data[key]
                if hasattr(mat, 'shape'):
                    constraint_matrix = mat
                    break

        # Analyze sparsity of the main constraint matrix
        if constraint_matrix is not None:
            total_entries = constraint_matrix.shape[0] * constraint_matrix.shape[1]
            nnz = None  # number of non-zeros
            if hasattr(constraint_matrix, 'nnz'):
                nnz = constraint_matrix.nnz
            elif hasattr(constraint_matrix, 'count_nonzero'):
                nnz = constraint_matrix.count_nonzero()
            else:
                try:
                    nnz = np.count_nonzero(constraint_matrix)
                except Exception as e:
                    try:
                        nnz = np.count_nonzero(constraint_matrix.toarray())
                    except Exception as e2:
                        pass

            # these numbers are calculated based on the solver canonical form
            self.solution_metrics['constraint_matrix_density'] = nnz / total_entries if total_entries > 0 else 0
            self.solution_metrics['constraint_matrix_sparsity'] = 1 - (nnz / total_entries) if total_entries > 0 else 0

        print(self.solution_metrics)

    def formulate_sparse_opt_problem(self):
        """
        Not in use
        This was a try to write efficient formulation with minimal dimension of z1
        The aim was to generate 1D variable and then multiply it by a matrix to arrange z1 in the desired 2d form
        The run times were extremely long and the issue of dimensionality solved with formulate_opt_problem
        based on cvxpy sparsity attribute
        :return:
        """
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()

        nonanticipative_mat = self.build_nonanticipative_matrix()
        row_indices, col_indices = np.where(nonanticipative_mat)
        z1_dim = len(row_indices)
        z1_var = cp.Variable(z1_dim)
        self.z0 = cp.Variable(self.n_indep_per_t * self.t)

        positions = np.where(nonanticipative_mat == 1)
        positions = list(zip(positions[0], positions[1]))
        A = np.zeros((nonanticipative_mat.shape[0] * nonanticipative_mat.shape[1], z1_dim))
        for idx, pos in enumerate(positions):
            flat_index = pos[0] * nonanticipative_mat.shape[1] + pos[1]  # Convert 2D index to 1D
            A[flat_index, idx] = 1

        self.z1 = cp.reshape(A @ z1_var, (nonanticipative_mat.shape[0], nonanticipative_mat.shape[1]), order='C')
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)
        w0 = sum(
            (self.B[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
            for t in range(self.t))
        w1 = sum((self.B[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                 + (self.B[:, self.t_cols[t]] @ self.k2[t])
                 @ self.z1[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map
                 for t in range(self.t))

        for j in range(self.B.shape[0]):
            self.constraints.append(-self.c[j] + w0[j] + (w1 @ self.b)[j]
                                    + self.omega_param * cp.norm((w1 @ self.projected_delta)[j], 2) <= 0)

        # formulate the objective function as constraint
        f = self.piecewise_costs.reshape(1, -1)
        Obj0 = sum((f[:, self.t_cols[t]] @ self.k2[t]) @ self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)]
                   for t in range(self.t))
        Obj1 = sum((f[:, self.t_cols[t]] @ self.k1[t] @ np.eye(len(self.b))[self.t_eq_rows[t]])
                   + (f[:, self.t_cols[t]] @ self.k2[t])
                   @ self.z1[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map
                   for t in range(self.t))
        self.constraints.append(-self.obj + Obj0 + Obj1 @ self.b
                                + self.omega_param * cp.norm(Obj1 @ self.projected_delta, 2) <= 0)

        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def read_solution(self, sol_path):
        with open(sol_path, "rb") as f:
            solution = pickle.load(f)
            self.z0 = solution['z0']
            self.z1 = solution['z1']

    def experimental_opt_formulation(self):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.t_formulation_start = time.time()
        self.z0 = cp.Variable(self.n_indep_per_t * self.t)
        self.z1 = cp.Variable((self.n_indep_per_t * self.t, self.n_uncertain_per_t * self.t))
        self.obj = cp.Variable(1)
        self.omega_param = cp.Parameter(nonneg=True)

        nonanticipative_mat = self.build_nonanticipative_matrix()
        # flip nonanticipative mat - constraint is on the elements not included
        nonanticipative_mat = 1 - nonanticipative_mat
        self.constraints += [cp.multiply(self.z1, nonanticipative_mat) == 0]

        w0 = (self.B @ self._k) @ self.z0
        w1 = self.B @ self._p2 @ np.linalg.pinv(self._A1) + self.B @ self._k @ self.z1 @ self.z_to_b_map
        for j in range(self.B.shape[0]):
            self.constraints.append(-self.c[j] + w0[j] + (w1 @ self.b)[j]
                                    + self.omega_param * cp.norm((w1 @ self.projected_delta)[j], 2) <= 0)

        f = self.piecewise_costs.reshape(1, -1)
        Obj0 = f @ self._k @ self.z0
        Obj1 = f @ (self._p2 @ np.linalg.pinv(self._A1)) + f @ self._k @ self.z1 @ self.z_to_b_map
        self.constraints.append(-self.obj + Obj0 + Obj1 @ self.b
                                + self.omega_param * cp.norm((Obj1 @ self.projected_delta), 2) <= 0)
        self.problem = cp.Problem(cp.Minimize(self.obj), constraints=self.constraints)

    def extract_solution_experimental_opt_formulation(self):
        if self.z1 is None:
            z1 = np.zeros((self.n_indep_per_t * self.t, self.b.shape[0]))
        else:
            z1 = self.z1.value

        y = self.z0.value + z1 @ self.b
        x = self._p1 @ y + (self._p2 @ (np.linalg.pinv(self._A1) @ (self.b - self._A2 @ y)))
        n_vars_per_t = len(self.t_cols[0]) - 1
        stacked_x = np.stack([x[n_vars_per_t * _: n_vars_per_t * (_ + 1)] for _ in range(self.t)])
        g = graphs.OptGraphs(self, stacked_x)
        g.plot_generators()
        g.tanks_volume()

    def extract_solution(self):
        # nominal solution
        yt = {t: self.z0[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)].value
                 + self.z1_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map @ self.b
              for t in range(self.t)}

        stacked_y = np.stack([_.T for _ in list(yt.values())])
        xt = {t: self.k1[t] @ self.b[self.t_eq_rows[t]] + self.k2[t] @ yt[t] for t in range(self.t)}

        stacked_x = np.stack([_.T for _ in list(xt.values())], axis=-1)
        self.x = stacked_x[:-self.n_gen].flatten('F')
        if self.n_bat_vars == 2:
            idx = ([f"bus {_}" for _ in range(self.n_bus)]
                   + [f"gen {_}" for _ in range(self.n_gen)]
                   + [f"bat-c {_}" for _ in range(self.n_bat)]
                   + [f"bat-d {_}" for _ in range(self.n_bat)]
                   + [f"comb {_}" for _ in range(self.n_combs)]
                   + [f"desal {_}" for _ in range(self.n_desal)]
                   + [f"tank {_}" for _ in range(self.n_tanks)]
                   + [f"w {_}" for _ in range(self.n_gen)]
                   )
        else:
            idx = ([f"bus {_}" for _ in range(self.n_bus)]
                   + [f"gen {_}" for _ in range(self.n_gen)]
                   + [f"bat-c {_}" for _ in range(self.n_bat)]
                   + [f"comb {_}" for _ in range(self.n_combs)]
                   + [f"desal {_}" for _ in range(self.n_desal)]
                   + [f"tank {_}" for _ in range(self.n_tanks)]
                   + [f"w {_}" for _ in range(self.n_gen)]
                   )
        df = pd.DataFrame(stacked_x, index=idx)
        # x = np.hstack([stacked_x[:-self.n_gen].flatten('F'), stacked_x[-self.n_gen:].flatten('F')])
        # df = pd.DataFrame({'Ax': self.A @ x, 'b': self.b})

    def switch_leading_index(self, sample, k, n):
        matrix_reshaped = np.reshape(sample, (k, self.t, n))
        matrix_transposed = np.transpose(matrix_reshaped, (1, 0, 2))
        return np.reshape(matrix_transposed, (self.t * k, n))

    def analyze_solution(self, n):
        def projected_cov():
            """
            Generate projected cov.
            Unlike the projection before the optimization this projection is used for random sample
            The optimization projection includes (1) convert from the cov temporal structure to bus-wise structure
            and (2) sum rows for buses with more than one uncertainty (loads and PV)
            The projection here includes only (1) - for the sample generation we want to draw the PV explicitly and
            not subtract it from the bus loads
            :return:
            """
            cov_size = (self.n_bus + self.n_bus + self.n_tanks) * self.t
            mat = np.zeros((cov_size, cov_size))
            for t in range(self.t):
                e = np.eye(self.t)[t]
                for b in range(self.n_bus):
                    mat[self.n_bus * t + b, b * self.t: (b + 1) * self.t] = e
                    mat[self.n_bus * self.t + self.n_bus * t + b,
                    self.n_bus * self.t + b * self.t: self.n_bus * self.t + (b + 1) * self.t] = e

            r = c = self.n_bus * self.t * 2
            for t in range(self.t):
                e = np.eye(self.t)[t]
                for tank_idx in range(self.n_tanks):
                    mat[r + self.n_tanks * t + tank_idx, c + tank_idx * self.t: c + (tank_idx + 1) * self.t] = e

            return mat @ self.cov @ mat

        # loads, injections, demands = self.get_nominal_rhs()
        # mu = np.hstack([loads, injections, demands])
        # zero_idx = np.where(mu == 0)[0]
        # cov_prime = projected_cov()
        # sample = uncertainty.draw_multivariate(mu=mu, cov=cov_prime, n=n)
        # sample[zero_idx, :] = 0
        #
        # sample_loads = sample[:self.n_bus * self.t, :]
        # sample_pv = sample[self.n_bus * self.t: self.n_bus * self.t * 2, :]
        # sample_dem = sample[self.n_bus * self.t * 2:, :]
        # sample = np.vstack([sample_loads - sample_pv, sample_dem])

        loads = self.pds.dem_active.values[:, :self.t].flatten()
        injections = np.multiply(self.pds.bus['max_pv_pu'].values, self.pds.max_gen_profile.values[:, :self.t].T).T.flatten()
        demands = self.wds.demands.values[:self.t, :].flatten('F')
        mu = np.hstack([loads, injections, demands])
        sample = uncertainty.draw_multivariate(mu=mu, cov=self.cov, n=n)
        sample[sample < 0] = 0
        sample_loads = self.switch_leading_index(sample[:self.n_bus * self.t, :], self.n_bus, n=n)
        sample_pv = self.switch_leading_index(sample[self.n_bus * self.t: 2 * self.n_bus * self.t, :], self.n_bus, n=n)
        sample_dem = self.switch_leading_index(sample[2 * self.n_bus * self.t:, :], self.n_tanks, n=n)
        sample = np.vstack([sample_loads - sample_pv, sample_dem])


        # old approach - for formulation absed on time step decomposition
        # yt = {t: self.z0_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)].reshape(-1, 1)
        #          + self.z1_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map @ sample
        #       for t in range(self.t)}
        # xt = {t: self.k1[t] @ sample[self.t_eq_rows[t]] + self.k2[t] @ yt[t] for t in range(self.t)}
        # self.x_by_sample = np.stack([_.T for _ in list(xt.values())], axis=-1)  # shape = (n, all_variables, T)

        y = self.z0_val.reshape(-1, 1) + (self.z1_val @ self.z_to_b_map) @ sample
        x = (self._k1 @ sample + self._k2 @ y).T

        # the total number of variables per time step
        n_vars_per_t = self.n_indep_per_t + self.n_dep_per_t
        x_vars = x[:, :(n_vars_per_t - self.n_gen) * self.t].reshape(n, self.t, (n_vars_per_t - self.n_gen))
        pw_vars = x[:, (n_vars_per_t - self.n_gen) * self.t:].reshape(n, self.t, self.n_gen)
        self.x_by_sample = np.concatenate([x_vars, pw_vars], axis=-1)
        self.x_by_sample = np.swapaxes(self.x_by_sample, 1, 2)
        costs = self.calculate_piecewise_linear_result(self.x_by_sample)
        print(f"Objective (AVG): {costs.sum(axis=(1, 2)).mean()}")

        violations = pd.DataFrame()
        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            x_tank_in = self.x_by_sample[:, self.n_pds + self.n_combs + self.n_desal + tank_idx]
            vol = np.hstack([np.tile([tank_data['init_vol']], (n, 1)),
                             (tank_data['init_vol'] + np.tril(np.ones((self.t, self.t))) @ x_tank_in.T).T])

            tank_ub = np.tile(tank_data['max_vol'], self.t + 1)
            tank_lb = np.tile(tank_data['min_vol'], self.t + 1)
            tank_lb[-1] = tank_data['init_vol']
            tank_violations = np.logical_or(np.any(vol - tank_lb < 0, axis=1), np.any(tank_ub - vol < 0, axis=1))
            violations[tank_name] = tank_violations

        for bat_idx, (bat_name, bat_data) in enumerate(self.pds.batteries.iterrows()):
            x_bat_in = (self.x_by_sample[:, self.pds.n_bus + self.pds.n_generators + bat_idx]
                        - self.x_by_sample[:, self.pds.n_bus + self.pds.n_generators + self.pds.n_batteries + bat_idx])
            x_bat_in = np.where(x_bat_in < 0, x_bat_in * (1 / bat_data["charge_eff"]),
                                x_bat_in * bat_data["discharge_eff"])
            x_bat_in *= self.pds.pu_to_mw

            gen_idx = [self.n_bus + _ for _ in range(self.n_gen)]
            desal_idx = [self.n_pds + self.n_combs + _ for _ in range(self.n_desal)]
            combs_idx = [self.n_pds + _ for _ in range(self.n_combs)]
            combs_power = self.wds.combs['total_power'].values
            combs_power = combs_power.reshape(1, len(combs_power), 1)
            pumps_power = self.x_by_sample[:, combs_idx, :] * combs_power * self.wds_power_units_factor
            bus_power_consumption = (sample_loads - sample_pv).T.reshape(n, self.t, self.n_bus).sum(axis=2)
            x_bat_in = (self.x_by_sample[:, gen_idx, :].sum(axis=1)
                        - pumps_power.sum(axis=1)
                        - (self.x_by_sample[:, desal_idx, :] * self.wds.desal_power * self.wds_power_units_factor).sum(
                        axis=1)
                        - bus_power_consumption)
            if self.n_bat_vars == 2:
                x_bat_in = np.where(x_bat_in < 0, x_bat_in * (1 / bat_data["charge_eff"]),
                                    x_bat_in * bat_data["discharge_eff"])
            x_bat_in *= self.pds.pu_to_mw

            init_soc = np.tile(bat_data['init_storage'], x_bat_in.shape[0]).reshape(-1, 1)
            soc = np.hstack([init_soc, (bat_data['init_storage'] + np.tril(np.ones((self.t, self.t))) @ x_bat_in.T).T])
            bat_ub = np.tile(bat_data['max_storage'], self.t + 1)
            bat_lb = np.tile(bat_data['min_storage'], self.t + 1)
            bat_lb[-1] = bat_data['init_storage'] - self.EPSILON
            bat_violations = np.logical_or(np.any(soc - bat_lb < 0, axis=1), np.any(bat_ub - soc < 0, axis=1))
            violations[bat_name] = bat_violations

        violations['total'] = violations.any(axis=1)
        if self.plot:
            g = graphs.OptGraphs(self, self.x_by_sample)
            g.tanks_volume()
            g.soc(soc_to_plot=soc)

        wc_cost = self.problem.value * self.piecewise_coef_scale
        avg_cost, max_cost = costs.sum(axis=(1, 2)).mean(), costs.sum(axis=(1, 2)).max()
        reliability = violations['total'].sum() / n
        return wc_cost, avg_cost, max_cost, reliability

    def calculate_piecewise_linear_result(self, x_values):
        # Ensure input is a numpy array for vectorized operations
        x_values = np.array(x_values, dtype=float)
        res = np.zeros((x_values.shape[0], self.n_gen, self.t))
        for i, (gen_idx, row) in enumerate(self.pds.generators.iterrows()):
            gen_x_values = x_values[:, self.n_bus + i, :] * self.pds.pu_to_mw
            bp = utils.quad_to_piecewise(a=row['gen_a'], b=row['gen_b'], c=row['gen_c'], p_min=row['min_gen_p'],
                                         p_max=row['max_gen_p'], num_segments=self.pw_segments)

            # Sort breakpoints and convert to numpy array
            x_points, y_points = zip(*bp)

            # Interpolation for values within the range
            y_values = np.interp(gen_x_values, x_points, y_points)

            # Handle extrapolation for values below the smallest x-point
            below_range = gen_x_values < x_points[0]
            if np.any(below_range):
                slope_below = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
                y_values[below_range] = y_points[0] + slope_below * (gen_x_values[below_range] - x_points[0])

            # Handle extrapolation for values above the largest x-point
            above_range = gen_x_values > x_points[-1]
            if np.any(above_range):
                slope_above = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
                y_values[above_range] = y_points[-1] + slope_above * (gen_x_values[above_range] - x_points[-1])

            res[:, i, :] = y_values
        return res