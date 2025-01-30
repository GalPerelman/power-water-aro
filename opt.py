import os
import pickle
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
import sympy as sp
import scipy
import copy
import seaborn as sns

import gurobipy as gp
from gurobipy import GRB

import graphs
import utils
import uncertainty
from pds import PDS
from wds import WDS

np.set_printoptions(linewidth=10 ** 5)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

class BaseOptModel:
    def __init__(self, pds, wds, t, omega=None, pw_segments=None, solver_params=None, solver_display=False):
        self.pds = pds
        self.wds = wds
        self.t = t
        self.omega = omega
        self.pw_segments = pw_segments
        self.solver_params = solver_params
        self.solver_display = solver_display

        self.wds_power_units_factor = (utils.POWER_UNITS[self.wds.input_power_units]
                                       / utils.POWER_UNITS[self.pds.input_power_units]) * self.pds.to_pu

        # for matrices dimensions
        self.n_bus, self.n_gen, self.n_bat = self.pds.n_bus, self.pds.n_generators, self.pds.n_batteries
        self.n_combs, self.n_desal, self.n_tanks = self.wds.n_combs, self.wds.n_desal, self.wds.n_tanks

        # total number of variables per time step (not including piecewise linear)
        # batteries have 2 variables (charging and discharging)
        self.n_pds, self.n_wds = self.n_bus + self.n_gen + 2 * self.n_bat, self.n_combs + self.n_desal + self.n_tanks
        self.n_tot = self.n_bus + self.n_gen + 2 * self.n_bat + self.n_combs + self.n_desal + self.n_tanks
        self.w = 0 if self.pw_segments is None else self.pw_segments + 1  # number of segments edges

        # piecewise variables
        self.n_pw_vars = self.n_gen * self.t
        self.piecewise_costs = np.zeros(self.n_tot * self.t + self.n_pw_vars)

        # inequality constraints are saved in dicts {name: mat1, name: mat2..], [name: b1, name:b2..]
        self.ineq_mat, self.ineq_rhs = {}, {}
        # equality constraints only one mat and rhs for the variables decomposition
        self.eq_mat, self.eq_rhs = self.build_equality_constraints()

        self.build_inequality_constraints()

        self.model = None
        self.constraints = []

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
        bat_cols = utils.get_mat_for_type(self.pds.bus, self.pds.batteries)
        bat_cols = bat_cols[:, np.any(bat_cols, axis=0)]
        charge_eff = self.pds.bus["charge_eff"].values.reshape(-1, 1)
        bat_charge_cols = np.divide(bat_cols, charge_eff, out=np.full_like(bat_cols, 0, dtype=float),
                                    where=charge_eff != 0)
        bat_charge_cols[np.abs(bat_charge_cols) < 10 ** -6] = 0

        discharge_eff = self.pds.bus["discharge_eff"].values.reshape(-1, 1)
        bat_discharge_cols = np.multiply(bat_cols, discharge_eff)
        bat_discharge_cols[np.abs(bat_discharge_cols) < 10 ** -6] = 0

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
            mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + (self.n_bus + self.n_gen):
                self.n_tot * t + (self.n_bus + self.n_gen) + self.n_bat] = -1 * bat_charge_cols
            mat[self.n_bus * t: self.n_bus * (t + 1),
                self.n_tot * t + (self.n_bus + self.n_gen + self.n_bat): self.n_tot * t + self.n_pds] \
                = bat_discharge_cols

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

        ub = np.tile(
            np.hstack([bus_angles_ub,  # upper bound for bus angles
                       self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'max_gen_p_pu'],  # generators ub
                       self.pds.bus.loc[self.pds.bus['max_charge'] > 0, 'max_charge'],  # batteries charge ub
                       self.pds.bus.loc[self.pds.bus['max_discharge'] > 0, 'max_discharge'],  # batteries discharge ub
                       np.tile(1, self.n_combs),  # upper bound for combs duration
                       self.wds.desal['max_flow'].values,  # desalination upper bound
                       np.tile(10 ** 6, self.n_tanks)  # tank inflows
                       ]), self.t)
        lb = np.tile(
            np.hstack([bus_angles_lb,  # lower bound for bus angles
                       self.pds.bus.loc[self.pds.bus['type'] == 'gen', 'min_gen_p_pu'],  # generators lb
                       np.tile(0, self.n_bat),  # lower bound for batteries charge
                       np.tile(0, self.n_bat),  # lower bound for batteries discharge
                       np.tile(0, self.n_combs),  # lower bound for combs duration
                       self.wds.desal['min_flow'].values,  # desalination lower bound
                       - np.tile(10 ** 6, self.n_tanks)  # tank inflows
                       ]), self.t)

        if self.pw_segments is not None:
            pw_ub = 10 ** 8  # inf
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
            bp = [(_[0] * self.pds.to_pu, _[1]) for _ in bp]

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
                charge_col = self.n_tot * t + (self.n_bus + self.n_gen) + b
                discharge_col = self.n_tot * t + (self.n_bus + self.n_gen + self.n_bat) + b
                start_row, end_row = 2 * b * self.t + t, (2 * b + 1) * self.t

                # first t rows are used for lb for the 'b' battery
                mat[start_row: end_row, charge_col] = -1
                mat[start_row: end_row, discharge_col] = 1
                # next block of t rows are used for ub for the 'b' battery
                mat[start_row + self.t: end_row + self.t, charge_col] = 1
                mat[start_row + self.t: end_row + self.t, discharge_col] = -1

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
        counter = 0
        for i, station in enumerate(self.wds.combs["station"].unique()):
            k = len(self.wds.combs.loc[self.wds.combs["station"] == station])  # num combs for station
            for t in range(self.t):
                mat[i * self.t + t, self.n_tot * t + self.n_pds + counter: self.n_tot * t + self.n_pds + counter + k] = 1
            counter += k

        if counter > 0:
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

    def get_variables_to_eliminate(self, method='qr'):
        if method == 'rref':
            echelon, indep_idx = sp.Matrix(self.eq_mat).rref()
            dep_idx = np.setxor1d(np.arange(self.eq_mat.shape[1]), indep_idx)

        else:
            q, r, p = scipy.linalg.qr(self.eq_mat, pivoting=True)
            tol = 1e-10  # tolerance for detecting non-zero entries
            rank = np.sum(np.abs(np.diag(r)) > tol)
            independent_columns_indices = p[:rank]
            indep_idx = independent_columns_indices
            dep_idx = np.setxor1d(np.arange(self.eq_mat.shape[1]), indep_idx)

        indep_idx, dep_idx = sorted(list(indep_idx)), sorted(list(dep_idx))
        return indep_idx, dep_idx

    def solve(self):
        self.model = cp.Problem(cp.Minimize(self.piecewise_costs @ self.x), self.constraints)
        self.model.solve(solver=cp.GUROBI, reoptimize=True)
        obj, status, solver_time = self.model.value, self.model.status, self.model.solver_stats.solve_time
        return obj, status, solver_time

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
    def __init__(self, pds, wds, t, omega, pw_segments=None, solver_params=None, solver_display=False):
        super().__init__(pds, wds, t, omega, pw_segments, solver_params, solver_display)

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


class RODCPF(BaseOptModel):
    """
    variables are dependent on the equality constraints RHS
    x_indep = (A_indep)^-1 * (b - A_dep) * x_dep

    Therefore, before decalring x, the RHS is updated to include uncertainty
    """
    def __init__(self, pds, wds, t, omega, pw_segments=None, solver_params=None, solver_display=False):
        super().__init__(pds, wds, t, omega, pw_segments, solver_params, solver_display)

        self.indep_idx, self.dep_idx = self.get_variables_to_eliminate(method='qr')
        self.delta = uncertainty.affine_mat(self.pds, self.wds, self.t)
        self.uncertain_rhs = self.add_rhs_uncertainty()

        self.lb, self.ub = self.get_x_bounds()
        self.x = self.declare_variables()
        self.formulate()

    def add_rhs_uncertainty(self):
        k = self.n_bus * self.t  # number of bus power balance constraints
        load_norm_terms = [self.omega * cp.norm(self.delta[i], 2) for i in range(k)]
        pv_norm_terms = [self.omega * cp.norm(self.delta[i + k], 2) for i in range(k)]

        stacked_rhs = cp.vstack([self.eq_rhs[i] + load_norm_terms[i] - pv_norm_terms[i] for i in range(k)])
        stacked_rhs = cp.vstack([stacked_rhs, np.zeros((self.t, 1))])  # add zeros for the ref bus = 0 constraints
        uncertain_rhs = cp.reshape(stacked_rhs, (self.eq_rhs.shape[0],))
        return uncertain_rhs

    def declare_variables(self):
        _x = cp.Variable(len(self.dep_idx))
        # extend _x to an expression includes all the decision variables (dependent and independent)
        p1 = np.zeros((self.eq_mat.shape[1], len(self.indep_idx)))
        p1[self.indep_idx, :] = np.eye(len(self.indep_idx))
        p2 = np.zeros((self.eq_mat.shape[1], len(self.dep_idx)))
        p2[self.dep_idx, :] = np.eye(len(self.dep_idx))
        x = (p1 @ np.linalg.pinv(self.eq_mat[:, self.indep_idx])
                  @ (self.uncertain_rhs - self.eq_mat[:, self.dep_idx] @ _x) + p2 @ _x)

        self.constraints += [x <= self.ub, x >= self.lb]
        return x

    def formulate(self):
        for (name_mat, mat), (name_rhs, rhs) in zip(self.ineq_mat.items(), self.ineq_rhs.items()):
            self.constraints.append(mat @ self.x <= rhs)


