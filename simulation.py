import math
import os.path

import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import ticker as mtick
import numpy as np
import pandas as pd

import graphs
import opt
import uncertainty
import utils

np.set_printoptions(linewidth=10 ** 5)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Simulation(opt.RobustModel):
    def __init__(self, pds_path, wds_path, t, omega, opt_method, elimination_method, manual_indep_variables,
                 z0, z1, plot, pw_segments=None, n_bat_vars=1, solver_params=None, solver_display=False,
                 # simulation arguments
                 n=1000, sample=None, **kwargs):
        super().__init__(pds_path, wds_path, t, omega, opt_method, elimination_method, manual_indep_variables,
                         pw_segments, n_bat_vars, solver_params, solver_display)

        self.z0_val = z0
        self.z1_val = z1
        self.n = n
        self.plot = plot
        self.sample = sample
        self.costs = None

        # OLD
        # mu = np.hstack(self.get_nominal_rhs())
        # cov = self.projected_cov()
        # self.sample = uncertainty.draw_multivariate(mu=mu, cov=cov, n=self.n)
        # self.sample_loads = self.sample[:self.n_bus * self.t, :]
        # self.sample_pv = self.sample[self.n_bus * self.t: self.n_bus * self.t * 2, :]
        # self.sample_dem = self.sample[self.n_bus * self.t * 2:, :]

        # get the nominal values - arranged according to the projected cov rows!
        # This means mu is not like the opt RHS
        if sample is None:
            loads = self.pds.dem_active.values[:, :self.t].flatten()
            injections = np.multiply(self.pds.bus['max_pv_pu'].values,
                                     self.pds.max_gen_profile.values[:, :self.t].T).T.flatten()
            demands = self.wds.demands.values[:self.t, :].flatten('F')
            mu = np.hstack([loads, injections, demands])
            self.sample = uncertainty.draw_multivariate(mu=mu, cov=self.cov, n=self.n)
        else:
            self.sample = sample

        self.sample[self.sample < 0] = 0
        self.sample_loads = self.switch_leading_index(self.sample[:self.n_bus * self.t, :], self.n_bus, n=self.n)
        self.sample_pv = self.switch_leading_index(self.sample[self.n_bus * self.t: 2 * self.n_bus * self.t, :],
                                                   self.n_bus, n=self.n)
        self.sample_dem = self.switch_leading_index(self.sample[2 * self.n_bus * self.t:, :], self.n_tanks, n=self.n)

        self.sample_rhs = np.vstack([self.sample_loads - self.sample_pv, self.sample_dem])

        self.x_nominal, self.x_by_sample = self.extract_x()
        self.graphs = graphs.OptGraphs(self, self.x_by_sample)
        self.solution = {"tanks": {}, "batteries": {}}
        self.violation_counts = pd.DataFrame()
        self.violation_volume = pd.DataFrame()
        self.total_violation_volume = pd.DataFrame()
        self.violation_dispatch = pd.DataFrame()
        self.violations_penalty = {}

    def projected_cov(self):
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

        row0 = self.n_bus * self.t * 2
        col0 = self.n_bus * self.t * 2
        for t in range(self.t):
            e = np.eye(self.t)[t]
            for tank_idx in range(self.n_tanks):
                mat[row0 + self.n_tanks * t + tank_idx, col0 + tank_idx * self.t: col0 + (tank_idx + 1) * self.t] = e

        cov_prime = mat @ self.cov @ mat
        return cov_prime

    def extract_x(self):
        # old approach - for formulation based on time step decomposition
        # yt = {t: self.z0_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)].reshape(-1, 1)
        #          + self.z1_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :] @ self.z_to_b_map @ self.sample_rhs
        #       for t in range(self.t)}
        # xt = {t: self.k1[t] @ self.sample_rhs[self.t_eq_rows[t]] + self.k2[t] @ yt[t] for t in range(self.t)}
        # x_by_sample = np.stack([_.T for _ in list(xt.values())], axis=-1)  # shape = (n, all_variables, T)

        y = self.z0_val.reshape(-1, 1) + (self.z1_val @ self.z_to_b_map) @ self.sample_rhs
        x = (self._k1 @ self.sample_rhs + self._k2 @ y).T
        n_vars_per_t = self.n_indep_per_t + self.n_dep_per_t
        x_vars = x[:, :(n_vars_per_t - self.n_gen) * self.t].reshape(self.n, self.t, (n_vars_per_t - self.n_gen))
        pw_vars = x[:, (n_vars_per_t - self.n_gen) * self.t:].reshape(self.n, self.t, self.n_gen)
        x_by_sample = np.concatenate([x_vars, pw_vars], axis=-1)
        x_by_sample = np.swapaxes(x_by_sample, 1, 2)

        # nominal solution
        yt = {t: self.z0_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1)].reshape(-1, 1)
                 + self.z1_val[self.n_indep_per_t * t: self.n_indep_per_t * (t + 1), :]
                 @ self.z_to_b_map @ self.b.reshape(-1, 1)
              for t in range(self.t)}
        xt = {t: self.k1[t] @ self.b[self.t_eq_rows[t]].reshape(-1, 1) + self.k2[t] @ yt[t] for t in range(self.t)}
        x_nominal = np.stack([_.T for _ in list(xt.values())], axis=-1)
        return x_nominal, x_by_sample

    def calculate_tank_volumes(self):
        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            flow_vars = self.x_by_sample[:, self.n_pds: self.n_pds + self.n_combs + self.n_desal, :]
            aux_arr = np.vstack([self.wds.combs['flow'].values.reshape(-1, 1), np.ones((self.n_desal, 1))])
            flows = aux_arr * flow_vars

            init_vol = self.wds.tanks.loc[tank_name, 'init_vol']
            vol = init_vol
            tank_idx = self.wds.tanks.index.get_loc(tank_name)

            demand = self.sample_dem.reshape(-1, self.n_tanks, self.n)[:, tank_idx, :]
            cumm_demand = demand.cumsum(axis=0).T
            for comb_idx, comb_data in self.wds.combs.iterrows():
                if comb_data['to'] == tank_name:
                    vol += flows[:, comb_idx, :].cumsum(axis=-1)
                if comb_data['from'] == tank_name:
                    vol -= flows[:, comb_idx, :].cumsum(axis=-1)

            for desal_idx, desal_data in self.wds.desal.iterrows():
                if desal_data['to'] == tank_name:
                    vol += flows[:, self.n_combs + desal_idx, :].cumsum(axis=-1)

            if vol.shape[0] == 1:
                vol = np.tile(vol, (self.n, 1)) - cumm_demand
            else:
                vol = vol - cumm_demand

            self.solution["tanks"][tank_name] = vol

            x_tank_in = self.x_nominal[:, self.n_pds + self.n_combs + self.n_desal + tank_idx, :]
            vol_by_tank_vars = (init_vol + np.tril(np.ones((self.t, self.t))) @ x_tank_in.T).T

            x_tank_in = self.x_by_sample[:, self.n_pds + self.n_combs + self.n_desal + tank_idx, :]
            tank_ub = np.tile(tank_data['max_vol'], self.t)
            tank_lb = np.tile(tank_data['min_vol'], self.t)
            tank_lb[-1] = tank_data['init_vol']
            tank_violations = np.logical_or(np.any(vol - tank_lb < 0, axis=1), np.any(tank_ub - vol < 0, axis=1))
            total_violated_volume = (np.sum(np.maximum(0, tank_lb - vol), axis=1)
                                     + np.sum(np.maximum(0, vol - tank_ub), axis=1))

            self.violation_counts[tank_name] = tank_violations
            self.violation_volume[tank_name] = total_violated_volume
            self.total_violation_volume[tank_name] = [total_violated_volume.sum()]
            """
            for case study I_3-bus_desalination_wds_aro:
            Desalination power = 3000 kw = 3 mw
            T1 average power for pumping = 0 water are supplied from the desalination plant
            T2 average power for pumping = 1300 kw = 1.3 mw
            total power (mw/m^3): T1=3, T2=4.3
            power cost by average of two generators
            gen_avg_power = 0.5 * (gen_min + gen_max) = 0.5 * (10 + 300) = 155 fpr both generators
            average cost = a * gen_avg_power ^ 2 + b * gen_avg_power + c
            average specific cost: gen1 = 3567.75 / 155 = 23, gen2 = 2828 / 155 = 18.24
            price per mw = 0.5 * (23 + 18.24) = 20.63
            price per m^3: T1 = 20.63 * 3 = 61.89, T2 = 20.63 * 4.3 = 88.71
            """
            penalty_cost = {'T1': 61.89, 'T2': 88.71}
            tank_shortage = vol - tank_lb
            tank_shortage[tank_shortage >= 0] = 0
            tank_surplus = tank_ub - vol
            tank_surplus[tank_surplus >= 0] = 0
            self.violations_penalty[tank_name] = (-tank_shortage.sum()) * penalty_cost[tank_name] * 1.2

    def calculate_soc(self):
        ncols = max(1, int(math.ceil(math.sqrt(self.pds.n_batteries))))
        nrows = max(1, int(math.ceil(self.pds.n_batteries / ncols)))

        if self.plot:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
            axes = np.atleast_2d(axes).ravel()

        for bat_idx, (bat_name, bat_data) in enumerate(self.pds.batteries.iterrows()):
            if not self.n_bus + self.n_gen + bat_idx in self.indep_idx:
                # if batteries are dependent variables - calculate SOC based on power (energy) balance
                print("Batteries are DEPENDENT variables")
                indep_gen_idx = [self.n_bus + _ for _ in range(self.n_gen) if self.n_bus + _ in self.indep_idx]
                dep_gen_idx = [self.n_bus + _ for _ in range(self.n_gen) if self.n_bus + _ not in indep_gen_idx]
                desal_idx = [self.n_pds + self.n_combs + _ for _ in range(self.n_desal)]
                combs_idx = [self.n_pds + _ for _ in range(self.n_combs)]
                combs_power = self.wds.combs['total_power'].values
                combs_power = combs_power.reshape(1, len(combs_power), 1)
                pumps_power = self.x_by_sample[:, combs_idx, :] * combs_power * self.wds_power_units_factor

                loads = self.sample_loads.reshape(self.t, self.n_bus, self.n).sum(axis=1).T
                pv = self.sample_pv.reshape(self.t, self.n_bus, self.n).sum(axis=1).T

                x_bat_in = (self.x_by_sample[:, indep_gen_idx, :].sum(axis=1)
                            + self.x_by_sample[:, dep_gen_idx, :].sum(axis=1)
                            - pumps_power.sum(axis=1)
                            - (self.x_by_sample[:, desal_idx, :] * self.wds.desal_power * self.wds_power_units_factor)
                            .sum(axis=1)
                            - loads + pv)
                if self.n_bat_vars == 2:
                    x_bat_in = np.where(x_bat_in < 0, x_bat_in * (1 / bat_data["charge_eff"]),
                                        x_bat_in * bat_data["discharge_eff"])
                x_bat_in *= self.pds.pu_to_mw

            if self.n_bus + self.n_gen + bat_idx in self.indep_idx:
                # if batteries are independent variables - calculate SOC based on the battery variables
                print("Batteries are INDEPENDENT variables")
                pass
                if self.n_bat_vars == 2:
                    x_bat_in = (self.x_by_sample[:, self.n_bus + self.n_gen + bat_idx, :]
                                - self.x_by_sample[:, self.n_bus + self.n_gen + self.n_bat + bat_idx, :]).copy()
                else:
                    x_bat_in = self.x_by_sample[:, self.n_bus + self.n_gen + bat_idx, :].copy()
                x_bat_in *= self.pds.pu_to_power_input_units

            init_soc = np.tile(bat_data['init_storage'], x_bat_in.shape[0]).reshape(-1, 1)
            soc = np.hstack([init_soc, (bat_data['init_storage'] + np.tril(np.ones((self.t, self.t))) @ x_bat_in.T).T])
            self.solution["batteries"][bat_name] = soc

            bat_ub = np.tile(bat_data['max_storage'], self.t + 1)
            bat_lb = np.tile(bat_data['min_storage'], self.t + 1)
            bat_lb[-1] = bat_data['init_storage']

            if not self.n_bus + self.n_gen + bat_idx in self.indep_idx:
                bat_violations = np.logical_or(np.any(soc - bat_lb + self.EPSILON < 0, axis=1),
                                               np.any(bat_ub - soc + self.EPSILON < 0, axis=1))
                self.violation_counts[f'B{bat_name}'] = bat_violations
            if self.n_bus + self.n_gen + bat_idx in self.indep_idx:
                # if batteries are independent variables - SOC violations never happens
                pass

            if self.plot:
                axes[bat_idx].plot(soc.T, 'C0', alpha=0.3)
                axes[bat_idx].hlines(bat_data['min_storage'], 0, self.t, 'r')
                axes[bat_idx].hlines(bat_data['max_storage'], 0, self.t, 'r')
                axes[bat_idx].hlines(bat_data['init_storage'], 0, self.t, 'k', linestyle='--')
                axes[bat_idx].grid()

        if self.plot:
            fig.text(0.5, 0.04, 'Time (hr)', ha='center')
            fig.text(0.04, 0.5, f'SOC ({self.pds.input_power_units.upper()}h)', va='center',
                     rotation='vertical')
            fig.subplots_adjust(bottom=0.14, top=0.95)

    def calculate_generators(self):
        for i, (gen_name, gen_data) in enumerate(self.pds.generators.iterrows()):

            # only for dependent generators:
            if not self.n_bus + i in self.indep_idx:
                gen_power = self.x_by_sample[:, self.n_bus + i, :] * self.pds.pu_to_mw
                gen_ub = np.tile(gen_data['max_gen_p'], self.t)
                gen_lb = np.tile(gen_data['min_gen_p'], self.t)
                gen_violations = np.logical_or(np.any(gen_power - gen_lb < 0, axis=1),
                                               np.any(gen_ub - gen_power < 0, axis=1))
                self.violation_counts[f'gen_{gen_name}'] = gen_violations

                total_violated_dispatch = (np.sum(np.maximum(0, gen_lb - gen_power), axis=1)
                                         + np.sum(np.maximum(0, gen_power - gen_ub), axis=1))

                self.violation_dispatch[gen_name] = total_violated_dispatch

    def calculate_costs(self):
        self.costs = self.calculate_piecewise_linear_result(self.x_by_sample).sum(axis=(1, 2))

    def plot_costs(self):
        fig, ax = plt.subplots()
        ax.hist(self.costs, bins=50, edgecolor='k')
        ax.set_xlabel("Cost ($)")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        # ax.ticklabel_format(style='plain', axis='x')
        ax.tick_params(axis='x', labelrotation=45)
        plt.subplots_adjust(bottom=0.25, top=0.96)

    def run(self):
        self.calculate_tank_volumes()
        self.calculate_soc()
        self.calculate_costs()
        self.calculate_generators()
        violations_rate = self.violation_counts.any(axis=1).sum() / self.n
        if self.plot:
            self.graphs.tanks_volume()
            self.graphs.plot_generators(shared_y=False)
        return self.costs, violations_rate


def simulate_experiment(experiment_path, thetas, export=False, analyze_lags=False, plot=False,
                        pds_latencies=None, wds_latencies=None):
    df = pd.DataFrame()
    for i, theta in enumerate(thetas):
        try:
            solution = utils.read_solution(sol_path=f"{experiment_path}_{theta}.pkl")
            sim = Simulation(**solution, plot=plot)
            costs, violations_rate = sim.run()

            violations_by_element = sim.violation_counts.sum(axis=0)
            sim_results = {"theta": theta, "avg_cost": costs.mean(), "max_cost": costs.max(), "std_cost": costs.std(),
                           "cost_skewness": scipy.stats.skew(costs),
                           "solver_obj": solution["solver_obj"],
                           "violations_rate": violations_rate,
                           "violations_penalty": sum(sim.violations_penalty.values()),
                           "total_violated_vol": sim.total_violation_volume.sum().sum(),
                           "total_violated_dispatch": sim.violation_dispatch.sum().sum(),
                           "pds_lat": sim.pds_lat, "wds_lat": sim.wds_lat}
            sim_results = {**sim_results, **violations_by_element}

            df = pd.concat([df, pd.DataFrame(sim_results, index=[len(df)])])
        except FileNotFoundError:
            pass

        pds_latencies = [0, 2, 4, 6, 8, 10, 12] if pds_latencies is None else pds_latencies
        wds_latencies = [0, 2, 4, 6, 8, 10, 12] if wds_latencies is None else wds_latencies
        if analyze_lags:
            for pds_lat in pds_latencies:
                for wds_lat in wds_latencies:
                    try:
                        sol_path = f"{experiment_path}_{theta}_pdslat-{pds_lat}_wdslat-{wds_lat}.pkl"
                        solution = utils.read_solution(sol_path=sol_path)
                        sim = Simulation(**solution, plot=plot)
                        costs, violations_rate = sim.run()

                        violations_by_element = sim.violation_counts.sum(axis=0)
                        sim_results = {"theta": theta, "avg_cost": costs.mean(), "max_cost": costs.max(),
                                       "std_cost": costs.std(),
                                       "cost_skewness": scipy.stats.skew(costs),
                                       "solver_obj": solution["solver_obj"],
                                       "violations_rate": violations_rate,
                                       "violations_penalty": sum(sim.violations_penalty.values()),
                                       "pds_lat": pds_lat, "wds_lat": wds_lat}
                        sim_results = {**sim_results, **violations_by_element}

                        df = pd.concat([df, pd.DataFrame(sim_results, index=[len(df)])])
                    except FileNotFoundError:
                        pass

    df["reliability"] = 1 - df["violations_rate"]
    df["penalized_avg_obj"] = df["avg_cost"] + df["violations_penalty"]
    print(df)
    if export:
        df.to_csv(f"{experiment_path}_por.csv")
    return df


def plot_advanced_por(experiment_path):
    ro = pd.DataFrame()
    aro = pd.DataFrame()
    thetas = [0.5, 1, 1.5, 2, 2.5, 3]

    fig, ax = plt.subplots()
    fig2, axes2 = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes2 = axes2.ravel()
    for i, theta in enumerate(thetas):
        try:
            solution = utils.read_solution(sol_path=os.path.join(aro_solution_dir, f"{experiment_name}_aro_{theta}.pkl"))
        except FileNotFoundError as e:
            print(e)
            continue
        sim = Simulation(**solution, plot=False)
        costs, violations_rate = sim.run()
        aro = pd.concat([aro, pd.DataFrame({"theta": theta, "avg_cost": costs.mean(), "max_cost": costs.max(),
                                            "solver_obj": solution["solver_obj"], "violations_rate": violations_rate,
                                            "pds_lat": sim.pds_lat, "wds_lat": sim.wds_lat, 'costs': [costs]},
                                           index=[len(aro)])])

        try:
            solution = utils.read_solution(sol_path=os.path.join(ro_solution_dir, f"{experiment_name}_ro_{theta}.pkl"))
        except FileNotFoundError as e:
            print(e)
            continue
        sim = Simulation(**solution, plot=False)
        costs, violations_rate = sim.run()
        ro = pd.concat([ro, pd.DataFrame({"theta": theta, "avg_cost": costs.mean(), "max_cost": costs.max(),
                                          "solver_obj": solution["solver_obj"], "violations_rate": violations_rate,
                                          "pds_lat": sim.pds_lat, "wds_lat": sim.wds_lat, 'costs': [costs]},
                                         index=[len(ro)])])

    ro["reliability"] = 1 - ro["violations_rate"]
    aro["reliability"] = 1 - aro["violations_rate"]

    ax_max = 1.15
    hist_ax_max = 1000
    x1, x2 = ro['reliability'].min(), ax_max
    y1, y2 = 0.05 * hist_ax_max, hist_ax_max
    m = (y2 - y1) / (x2 - x1)

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)  # Makes ax on top of ax2
    ax.patch.set_visible(False)

    ro_ax2 = axes2[0].twinx()
    aro_ax2 = axes2[1].twinx()
    axes2[0].set_zorder(ro_ax2.get_zorder() + 1)  # Makes ax on top of ax2
    axes2[0].patch.set_visible(False)
    axes2[1].set_zorder(aro_ax2.get_zorder() + 1)  # Makes ax on top of ax2
    axes2[1].patch.set_visible(False)

    for i, theta in enumerate([0.5, 1, 1.5, 3]):
        c = aro.loc[aro['theta'] == theta, 'costs'].values[0]
        r = aro.loc[aro['theta'] == theta, 'reliability'].values[0]

        counts, bins = np.histogram(c, bins=25)
        ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                bottom=m * (r - x1) + y1)
        label = 'ARO Costs' if i == 0 else '_nolegend_'
        aro_ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                bottom=m * (r - x1) + y1, label=label)

        c = ro.loc[aro['theta'] == theta, 'costs'].values[0]
        r = ro.loc[aro['theta'] == theta, 'reliability'].values[0]
        counts, bins = np.histogram(c, bins=25)
        label = 'RO Costs' if i == 0 else '_nolegend_'
        ro_ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                bottom=m * (r - x1) + y1, label=label)

    def customized_plot(ax, x, y, label, color):
        ax.plot(x, y, label=label, marker='o', zorder=10, c=color, mfc="w")
        return ax

    ax = customized_plot(ax, aro['avg_cost'], aro['reliability'], label='ARO-AVG', color=graphs.COLORS["ARO-AVG"])
    ax = customized_plot(ax, ro['avg_cost'], ro['reliability'], label='RO-AVG', color=graphs.COLORS["RO-AVG"])
    ax = customized_plot(ax, ro['max_cost'], ro['reliability'], label='RO-MAX', color=graphs.COLORS["RO-MAX"])
    ax2.set_ylim(0, hist_ax_max)
    ax2.set_yticks([])
    ax.set_ylim(0.9 * ro['reliability'].min(), ax_max)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    axes2[0] = customized_plot(axes2[0], ro['avg_cost'], ro['reliability'], label='RO-AVG', color=graphs.COLORS["RO-MAX"])
    axes2[1] = customized_plot(axes2[1], aro['avg_cost'], aro['reliability'], label='ARO-AVG', color=graphs.COLORS["ARO-AVG"])
    ro_ax2.set_ylim(0, hist_ax_max)
    ro_ax2.set_yticks([])
    aro_ax2.set_ylim(0, hist_ax_max)
    aro_ax2.set_yticks([])
    axes2[0].set_ylim(0.85 * ro['reliability'].min(), ax_max)
    axes2[1].set_ylim(0.85 * ro['reliability'].min(), ax_max)
    axes2[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    axes2[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    axes2[0].set_axisbelow(True)
    axes2[1].set_axisbelow(True)
    axes2[0].grid(color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    axes2[1].grid(color='gray', linewidth=0.5, linestyle='--', alpha=0.7)

    h1, lb1 = ro_ax2.get_legend_handles_labels()
    h2, lb2 = axes2[0].get_legend_handles_labels()
    axes2[0].legend(h1 + h2, lb1 + lb2, loc='lower right')
    h1, lb1 = aro_ax2.get_legend_handles_labels()
    h2, lb2 = axes2[1].get_legend_handles_labels()
    axes2[1].legend(h1 + h2, lb1 + lb2, loc='lower right')

    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("CSR (%)")
    ax.legend(loc='lower right')

    fig2.text(0.55, 0.03, 'Cost ($)', ha='center')
    fig2.text(0.025, 0.5, f'CSR (%)', va='center', rotation='vertical')

    fig.subplots_adjust(hspace=0, left=0.15, right=0.95, top=0.92, bottom=0.14)
    fig2.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.12, hspace=0.06)


def plot_ro_vs_aro(ro_path, aro_path, thetas):
    batteries_fig, batteries_ax = plt.subplots()

    for i, theta in enumerate(thetas):
        ro = utils.read_solution(sol_path=f"{ro_path}_{theta}.pkl")
        sim = Simulation(**ro, plot=False)
        costs, violations_rate = sim.run()
        tanks_fig = sim.graphs.tanks_volume(color="C1", leg_label="RO")
        generators_fig = sim.graphs.plot_generators(shared_y=False, color="C1", leg_label="RO", zo=5)
        for bat_idx, (bat_name, bat_data) in enumerate(sim.pds.batteries.iterrows()):
            batteries_ax.plot(sim.solution["batteries"][bat_name].T, 'C1', alpha=0.3, zorder=2)
            batteries_ax.grid(True)
            batteries_ax.plot(sim.solution["batteries"][bat_name].T[:, 0], 'C1', label="RO")

        aro = utils.read_solution(sol_path=f"{aro_path}_{theta}.pkl")
        sim = Simulation(**aro, plot=False)
        costs, violations_rate = sim.run()
        tanks_fig = sim.graphs.tanks_volume(fig=tanks_fig, color='C0', leg_label="ARO")
        generators_fig = sim.graphs.plot_generators(shared_y=False, fig=generators_fig, color="C0", leg_label="ARO",
                                                    zo=1)
        for bat_idx, (bat_name, bat_data) in enumerate(sim.pds.batteries.iterrows()):
            batteries_ax.plot(sim.solution["batteries"][bat_name].T, 'C0', alpha=0.3, zorder=5)
            batteries_ax.hlines(bat_data['min_storage'], 0, sim.t, 'r', zorder=10)
            batteries_ax.hlines(bat_data['max_storage'], 0, sim.t, 'r', zorder=10)
            batteries_ax.hlines(bat_data['init_storage'], 0, sim.t, 'k', linestyle='--', zorder=10)
            batteries_ax.grid(True)
            batteries_ax.plot(sim.solution["batteries"][bat_name].T[:, 0], 'C0', label="ARO")

        batteries_ax.legend(framealpha=1)
        batteries_ax.set_xlabel('Time (hr)')
        batteries_ax.set_ylabel(f'SOC ({sim.pds.input_power_units.upper()}h)')


if __name__ == "__main__":
    thetas = [0.5, 1, 1.5, 2, 2.5, 3]

    # simulate and plot case study I
    simulate_experiment("output/I_3-bus-desalination-wds_ro", thetas=thetas, export=True, plot=False)
    simulate_experiment("output/I_3-bus-desalination-wds_aro", thetas=thetas, export=True, plot=False,
                        analyze_lags=False, pds_latencies=None, wds_latencies=None)
    graphs.por(ro_results="output/I_3-bus-desalination-wds_ro_por.csv",
               aro_results="output/I_3-bus-desalination-wds_aro_por.csv")
    plot_advanced_por("output/I_3-bus-desalination-wds")

    # simulate and plot case study II
    simulate_experiment("output/II_ieee9-national-wds_ro", thetas, export=True, plot=False)
    simulate_experiment("output/II_ieee9-national-wds_aro", thetas, export=True, plot=False)
    graphs.por(ro_results="output/II_ieee9-national-wds_ro_por.csv",
               aro_results="output/II_ieee9-national-wds_aro_por.csv")
    plot_advanced_por("output/II_ieee9-national-wds")

    # latency analysis plot
    graphs.analyze_latency(aro_path="output/I_3-bus-desalination-wds_aro_por.csv",
                           ro_path="output/I_3-bus-desalination-wds_ro_por.csv")

    # plot decision space
    plot_ro_vs_aro("output/I_3-bus-desalination-wds_ro", aro_path="output/I_3-bus-desalination-wds_aro",
                   thetas=[1])
    plot_ro_vs_aro("output/I_sa_dep_bat_ro", aro_path="output/I_sa_dep_bat_aro",
                   thetas=[1])

    plt.show()