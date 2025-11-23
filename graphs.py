import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker as mtick
from matplotlib.ticker import ScalarFormatter, AutoLocator, Locator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COLORS = {"ARO": "#005d8f", "RED": "#d62828", "RO-MAX": "#f77f00", "RO-AVG": "#fcbf49"}
COLORS = {"ARO-AVG": "#1151b0", "ARO-MAX": "#1e5bba", "RED": "#d62828", "RO-MAX": "#e42727", "RO-AVG": "#f69c46"}


class OptGraphs:
    def __init__(self, model, x):
        self.model = model
        self.pds = model.pds
        self.wds = model.wds

        self.x = x
        self.alpha = 0.3

    def tanks_volume(self, fig=None, color='C0', leg_label="", tank_titles=True):
        n = self.wds.n_tanks
        ncols = max(1, int(math.ceil(math.sqrt(n))))
        nrows = max(1, int(math.ceil(n / ncols)))

        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(9, 4.5))
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            inflow = self.x[:, self.model.n_pds + self.wds.n_combs + self.wds.n_desal + tank_idx] * self.wds.flows_factor
            if inflow.shape[0] != 1:
                init_vol = np.tile(tank_data['init_vol'], inflow.shape[0]).reshape(-1, 1) * self.wds.flows_factor
            else:
                init_vol = np.array([tank_data['init_vol'] * self.wds.flows_factor]).reshape(-1, 1)

            axes[tank_idx].hlines(tank_data['min_vol'] * self.wds.flows_factor, 0, self.model.t, 'k', linewidth=1.5)
            axes[tank_idx].hlines(tank_data['max_vol'] * self.wds.flows_factor, 0, self.model.t, 'k', linewidth=1.5)
            axes[tank_idx].hlines(tank_data['init_vol'] * self.wds.flows_factor, 0, self.model.t, 'k', linestyle='--', zorder=20)

            y = np.hstack([init_vol, tank_data['init_vol'] * self.wds.flows_factor + inflow.cumsum(axis=-1)]).T
            axes[tank_idx].plot(y, color, alpha=self.alpha)
            # axes[tank_idx].scatter(0, tank_data['init_vol'] * self.wds.flows_factor, facecolor="none", edgecolor='r')
            axes[tank_idx].grid(True)
            if tank_titles:
                axes[tank_idx].set_title(f'Tank {tank_name}')

            if leg_label:
                axes[tank_idx].plot(y[:, 0], color, label=leg_label)
                axes[tank_idx].legend(framealpha=1)

        for j in (n, nrows * ncols + 1):
            try:
                axes[j].axis('off')
            except IndexError:
                pass
        fig.subplots_adjust(left=0.105, bottom=0.15, top=0.95, right=0.92, wspace=0.2)
        return fig

    def soc(self, soc_to_plot=None, fig=None, color="C0"):
        ncols = max(1, int(math.ceil(math.sqrt(self.pds.n_batteries))))
        nrows = max(1, int(math.ceil(self.pds.n_batteries / ncols)))

        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for bat_idx, (bat_name, bat_data) in enumerate(self.pds.batteries.iterrows()):
            x_bat_in = (self.x[:, self.pds.n_bus + self.pds.n_generators + bat_idx]
                        - self.x[:, self.pds.n_bus + self.pds.n_generators + self.pds.n_batteries + bat_idx])
            x_bat_in *= self.pds.pu_to_mw

            init_soc = np.tile(bat_data['init_storage'], x_bat_in.shape[0]).reshape(-1, 1)
            soc = np.hstack([init_soc, (bat_data['init_storage'] + np.tril(np.ones((self.model.t, self.model.t))) @ x_bat_in.T).T])
            if soc_to_plot is not None:
                axes[bat_idx].plot(soc_to_plot.T, color, alpha=self.alpha)

            axes[bat_idx].hlines(y=bat_data['min_storage'], xmin=0, xmax=self.model.t, color='r')
            axes[bat_idx].hlines(y=bat_data['max_storage'], xmin=0, xmax=self.model.t, color='r')
            axes[bat_idx].hlines(y=bat_data['init_storage'], xmin=0, xmax=self.model.t, color='k', linestyle='--')
            axes[bat_idx].grid()
        return fig

    def plot_generators(self, shared_y=False, fig=None, color="C0", leg_label="", zo=1, gen_titles=True):
        n = self.pds.n_generators
        if fig is None:
            ncols = max(1, math.ceil(math.sqrt(self.pds.n_generators)))
            nrows = max(1, int(math.ceil(self.pds.n_generators / ncols)))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=shared_y, figsize=(9, 4))
            axes = np.atleast_2d(axes).ravel()

            for j in (n, nrows * ncols + 1):
                try:
                    axes[j].axis('off')
                except IndexError:
                    pass

        else:
            axes = fig.axes

        for i, (gen_idx, gen_data) in enumerate(self.pds.generators.iterrows()):
            p = self.x[:, self.model.n_bus + i] * self.pds.pu_to_mw
            axes[i].plot(p.T, color, alpha=0.6, zorder=zo)
            axes[i].set_axisbelow(True)
            axes[i].grid(True, zorder=0)
            # axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # axes[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
            # axes[i].yaxis.set_major_locator(ConstantAwareLocator(tol=1e-10, fixed_interval=1, fixed_range=20))
            axes[i].yaxis.set_major_locator(ConstantAwareLocator(tol=0.1, width=3, step=1))
            axes[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if leg_label:
                axes[i].plot(p.T[:, 0], color, label=leg_label)
                leg = axes[i].legend(framealpha=1)
                leg.set_zorder(50)

            if gen_titles:
                axes[i].set_title(f'Generator {gen_idx + 1}')

        fig.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig.text(0.02, 0.55, f'Generation ({self.pds.input_power_units.upper()})', va='center',
                 rotation='vertical')

        fig.subplots_adjust(left=0.1, bottom=0.15, top=0.9, right=0.92, wspace=0.2)
        return fig

    def plot_pumping_stations(self):
        desal_idx = [self.model.n_pds + self.model.n_combs + _ for _ in range(self.model.n_desal)]
        combs_idx = [self.model.n_pds + _ for _ in range(self.model.n_combs)]
        combs_power = self.wds.combs['total_power'].values
        combs_power = combs_power.reshape(1, len(combs_power), 1)

        combs_flow = self.wds.combs['flow'].values
        combs_flow = combs_flow.reshape(1, len(combs_flow), 1)

        pumps_power = self.x[:, combs_idx, :] * combs_power * self.model.wds_power_units_factor
        pumps_flow = self.x[:, combs_idx, :] * combs_flow * self.model.wds.flows_factor

        ncols = max(1, math.ceil(math.sqrt(self.model.wds.n_stations)))
        nrows = max(1, int(math.ceil(self.model.wds.n_stations / ncols)))
        wds_pumping_stations = self.wds.combs['station'].unique()

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
        axes = np.atleast_2d(axes).ravel()
        for i, station_name in enumerate(wds_pumping_stations):
            station_idx = self.wds.combs.loc[self.wds.combs['station'] == station_name].index
            axes[i].plot(pumps_flow[:, station_idx, :].sum(axis=1).T, 'C0', alpha=self.alpha)
            axes[i].grid()

        fig.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig.text(0.02, 0.55, f'Pumping Station Flow ($m^3/hr$)', va='center', rotation='vertical')

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
        axes = np.atleast_2d(axes).ravel()
        for i, station_name in enumerate(wds_pumping_stations):
            station_idx = self.wds.combs.loc[self.wds.combs['station'] == station_name].index
            axes[i].plot(pumps_power[:, station_idx, :].sum(axis=1).T * self.pds.pu_to_power_input_units,
                         'C0', alpha=self.alpha)
            axes[i].grid()

        fig.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig.text(0.02, 0.55, f'Pumping Station Power ({self.pds.input_power_units.upper()})', va='center',
                 rotation='vertical')


class ConstantAwareLocator(Locator):
    """
    • If current span < tol  →  enlarge limits to [m-width, m+width]
      and return ticks every *step* in that band.
    • Else                  →  delegate to *fallback* (default AutoLocator).
    """
    def __init__(self, tol=1.0, width=5, step=1, fallback=None):
        super().__init__()
        self.tol = tol
        self.width = width
        self.step = step
        self.fallback = fallback or AutoLocator()
        self._fixed_limits = None        # remember whether we have expanded

    def set_axis(self, axis):
        super().set_axis(axis)
        if hasattr(self.fallback, "set_axis"):
            self.fallback.set_axis(axis)

    def __call__(self):
        if self.axis is None:
            return self.fallback()

        vmin, vmax = self.axis.get_view_interval()
        if abs(vmax - vmin) < self.tol:
            mid = 0.5 * (vmin + vmax)
            lo = np.floor((mid - self.width) / self.step) * self.step
            hi = np.ceil((mid + self.width) / self.step) * self.step

            if self._fixed_limits != (lo, hi):
                self.axis.axes.set_ylim(lo, hi)
                self._fixed_limits = (lo, hi)

            return np.arange(lo, hi + self.step, self.step)

        self._fixed_limits = None
        return self.fallback()

    def tick_values(self, vmin, vmax):
        return self.__call__()


def plot_mat(mat, norm=False, t=24):
    WhtBlRd = ["#ffffff", "#8ecae6", "#046B9F", "#fdf4b0", "#ffbf1f", "#b33005"]
    cmap = get_continuous_cmap(WhtBlRd)

    fig = plt.figure()
    if norm:
        mat = (mat - mat.min()) / (mat.max() - mat.min())

    mat_norm = max(abs(mat.min()), abs(mat.max()))
    im = plt.imshow(mat, cmap=cmap)
    ax = plt.gca()

    ax.tick_params(which='minor', bottom=False, left=False)
    cbar = plt.colorbar(im, ticks=mtick.AutoLocator())

    # Major ticks
    ax.set_xticks(np.arange(-0.5, mat.shape[0], t))
    ax.set_yticks(np.arange(-0.5, mat.shape[0], t))
    ax.set_xticklabels(np.arange(0, mat.shape[0] + t, t))
    ax.set_yticklabels(np.arange(0, mat.shape[0] + t, t))

    # Gridlines based on minor ticks
    ax.grid(which='major', color='k', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.4, alpha=0.4)

    ax.set_xticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    fig.subplots_adjust(top=0.9, bottom=0.13, left=0.055, right=0.9, hspace=0.2, wspace=0.2)
    return ax


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
        creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns color map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def por(ro_results="", aro_results=""):
    fig, ax = plt.subplots()
    if ro_results:
        ro = pd.read_csv(ro_results)
        ax.plot(ro['avg_cost'], ro['reliability'], label='RO-AVG', marker='o', mfc="w", c=COLORS["RO-AVG"])
    if aro_results:
        aro = pd.read_csv(aro_results, index_col=0)
        if 'pds_lat' in aro.columns or 'wds_lat' in aro.columns:
            aro = aro.loc[(aro['pds_lat'] == 0) & (aro['wds_lat'] == 0)]
            latency_results = aro[['theta', 'avg_cost', 'pds_lat', 'wds_lat', 'reliability']]
            latency_results = latency_results.loc[(latency_results['pds_lat'] > 0) & (latency_results['wds_lat'] > 0)]

        ax.plot(aro['avg_cost'], aro['reliability'], label='ARO-AVG', marker='o', mfc="w", c=COLORS["ARO-AVG"])

    ax.grid()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("CSR (%)")
    ax.legend(loc="lower right")


def advanced_por(experiment_name, ro_solution_dir, aro_solution_dir, ro_thetas_for_hist, aro_thetas_for_hist,
                 y_scale_param=0.01):
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
    hist_ax_max = 800
    x1, x2 = ro['reliability'].min(), ax_max
    y1, y2 = y_scale_param * hist_ax_max, hist_ax_max
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

    for i, theta in enumerate(aro_thetas_for_hist):
        c = aro.loc[aro['theta'] == theta, 'costs'].values[0]
        r = aro.loc[aro['theta'] == theta, 'reliability'].values[0]
        counts, bins = np.histogram(c, bins=30)
        ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                bottom=m * (r - x1) + y1)
        label = 'ARO Costs' if i == 0 else '_nolegend_'
        aro_ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                    bottom=m * (r - x1) + y1, label=label)

    for i, theta in enumerate(ro_thetas_for_hist):
        c = ro.loc[ro['theta'] == theta, 'costs'].values[0]
        r = ro.loc[ro['theta'] == theta, 'reliability'].values[0]
        counts, bins = np.histogram(c, bins=30)
        label = 'RO Costs' if i == 0 else '_nolegend_'
        ro_ax2.bar(bins[:-1], counts, width=np.diff(bins), color='lightgrey', edgecolor='k', alpha=0.5, zorder=1,
                   bottom=m * (r - x1) + y1, label=label)

    def customized_plot(ax, x, y, label, color):
        ax.plot(x, y, label=label, marker='o', zorder=10, c=color, mfc="w")
        return ax

    ax = customized_plot(ax, aro['avg_cost'], aro['reliability'], label='ARO-AVG', color=COLORS["ARO-AVG"])
    ax = customized_plot(ax, ro['avg_cost'], ro['reliability'], label='RO-AVG', color=COLORS["RO-AVG"])
    ax = customized_plot(ax, ro['max_cost'], ro['reliability'], label='RO-MAX', color=COLORS["RO-MAX"])
    ax2.set_ylim(0, hist_ax_max)
    ax2.set_yticks([])
    ax.set_ylim(0.9 * ro['reliability'].min(), ax_max)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    axes2[0] = customized_plot(axes2[0], ro['avg_cost'], ro['reliability'], label='RO-AVG', color=COLORS["RO-MAX"])
    axes2[1] = customized_plot(axes2[1], aro['avg_cost'], aro['reliability'], label='ARO-AVG', color=COLORS["ARO-AVG"])
    ro_ax2.set_ylim(0, hist_ax_max)
    ro_ax2.set_yticks([])
    aro_ax2.set_ylim(0, hist_ax_max)
    aro_ax2.set_yticks([])

    min_y = 0.85 * ro['reliability'].min() if 0.85 * ro['reliability'].min() > 0 else -0.02
    axes2[0].set_ylim(min_y, ax_max)
    axes2[1].set_ylim(min_y, ax_max)
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


def analyze_latency(aro_path, thetas):
    from scipy.interpolate import griddata
    df = pd.read_csv(aro_path, index_col=0)
    df = df.loc[df['theta'] != 0]

    # just for review the results
    df['penalty_deviation'] = df['penalized_avg_obj'] / df['avg_cost']
    df['penalty_cost'] = df['penalized_avg_obj'] - df['avg_cost']

    num_categories = len(np.unique(df['pds_lat']))
    cmap = plt.get_cmap('RdBu_r', num_categories)  # using a colormap with discrete colors
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
    c1 = axes[0].scatter(df['avg_cost'], df["reliability"], c=df['wds_lat'], cmap=cmap, edgecolor='k', alpha=0.8)
    c2 = axes[1].scatter(df['avg_cost'], df["reliability"], c=df['pds_lat'], cmap=cmap, edgecolor='k', alpha=0.8)

    fig.text(0.5, 0.04, 'Cost ($)', ha='center')
    fig.text(0.02, 0.55, f'CSR (%)', va='center', rotation='vertical')

    cax = inset_axes(axes[0], width="35%", height="5%", loc='lower right', bbox_to_anchor=(-0.08, 0.15, 1, 1),
                     bbox_transform=axes[0].transAxes, borderpad=0)
    cb = plt.colorbar(c1, cax=cax, orientation='horizontal')
    cb.set_label("WDS Latency (hr)")
    cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    cax = inset_axes(axes[1], width="35%", height="5%", loc='lower right',  bbox_to_anchor=(-0.08, 0.15, 1, 1),
                     bbox_transform=axes[1].transAxes, borderpad=0)
    cb = plt.colorbar(c2, cax=cax, orientation='horizontal')
    cb.set_label("PDS Latency (hr)")
    cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.96, top=0.95, wspace=0.1)
    #######################################################################################################

    fig, axes = plt.subplots(ncols=len(thetas), subplot_kw=dict(projection='3d'))
    axes = np.atleast_2d(axes).ravel()
    for i, _ in enumerate(thetas):
        temp = df.loc[df['theta'] == _]
        x = temp['pds_lat'].values
        y = temp['wds_lat'].values
        obj = temp['penalized_avg_obj'].values
        reliability = temp['reliability'].values

        grid_x, grid_y = np.mgrid[0:x.max():25j, 0:y.max():25j]
        points = np.array([x, y]).T
        grid = griddata(points, np.array(obj), (grid_x, grid_y), method='linear')

        mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
        mappable.set_array(grid)
        mappable.set_clim(obj.min(), obj.max())

        axes[i].plot_surface(grid_x, grid_y, grid, rstride=1, cstride=1, edgecolor='k', lw=0.5, cmap=plt.cm.RdBu_r, alpha=0.8)

        axes[i].zaxis.set_rotate_label(False)  # disable automatic rotation
        axes[i].set_zlabel('Constraint Penalized Cost ($)', rotation=90)
        axes[i].set_xlabel('PDS Latency (hr)')
        axes[i].set_ylabel('WDS latency (hr)')
        axes[i].zaxis.labelpad = 10
        axes[i].tick_params(axis='z', pad=8)
        axes[i].set_title(f'$\Omega$={_}', y=0.95)

        axes[i].azim = -130
        axes[i].elev = 15
    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.2)

    ############################################################
    fig, axes = plt.subplots(ncols=2, subplot_kw=dict(projection='3d'), figsize=(9, 4))
    temp = df.loc[df['theta'] == 1]
    x = temp['pds_lat'].values
    y = temp['wds_lat'].values
    obj = temp['avg_cost'].values
    reliability = temp['reliability'].values

    grid_x, grid_y = np.mgrid[0:x.max():25j, 0:y.max():25j]
    points = np.array([x, y]).T
    grid = griddata(points, np.array(obj), (grid_x, grid_y), method='linear')

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    mappable.set_array(grid)
    mappable.set_clim(obj.min(), obj.max())
    axes[0].plot_surface(grid_x, grid_y, grid, rstride=1, cstride=1, edgecolor='k', lw=0.5,cmap=plt.cm.RdBu_r, alpha=0.8)

    grid = griddata(points, np.array(reliability), (grid_x, grid_y), method='linear')
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    mappable.set_array(grid)
    mappable.set_clim(reliability.min(), reliability.max())
    axes[1].plot_surface(grid_x, grid_y, grid, rstride=1, cstride=1, edgecolor='k', lw=0.5, cmap=plt.cm.RdBu_r, alpha=0.8)

    for ax in axes:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_xlabel('PDS Latency (hr)')
        ax.set_ylabel('WDS latency (hr)')
        ax.azim = -40
        ax.elev = 15

    axes[0].text2D(0.05, 0.9, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top')
    axes[1].text2D(0.05, 0.9, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top')

    axes[0].tick_params(axis='z', pad=6)
    axes[1].tick_params(axis='z', pad=1)
    axes[0].zaxis.labelpad = 10
    axes[1].zaxis.labelpad = 5
    axes[0].set_zlabel('Cost ($)', rotation=90)
    axes[1].set_zlabel('CSR (%)', rotation=90)
    plt.subplots_adjust(left=0.04, right=0.9, wspace=0.1)


def plot_nonanticipative_matrix():
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

    t = 12
    n, k = 1, 2
    lat = 0
    lags = t
    mat = get_ldr_block(n=n, k=k, T=t, lags=lags, lat=lat)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    axes[0].imshow(get_ldr_block(n=n, k=k, T=t, lags=t, lat=0), cmap='Greys', interpolation='none')
    axes[1].imshow(get_ldr_block(n=n, k=k, T=t, lags=4, lat=0), cmap='Greys', interpolation='none')
    axes[2].imshow(get_ldr_block(n=n, k=k, T=t, lags=t, lat=4), cmap='Greys', interpolation='none')
    for ax in axes:
        minor_ticks_x = np.arange(-0.5, mat.shape[1], 1)
        minor_ticks_y = np.arange(-0.5, mat.shape[0], 1)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', left=False)
        ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)

        major_ticks_x = np.arange(0, mat.shape[1], 2)
        major_ticks_y = np.arange(0, mat.shape[0], 2)
        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)
        ax.set_xticklabels(major_ticks_x)
        ax.set_yticklabels(major_ticks_y)

    # legend_elements = [
    #     Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
    #            markeredgecolor='k', markersize=10, label='1'),
    #     Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
    #            markeredgecolor='k', markersize=10, label='0')
    # ]
    axes[0].text(x=0.01, y=1.06, s=f"(a)", fontsize=12, transform=axes[0].transAxes)
    axes[1].text(x=0.01, y=1.06, s=f"(b)", fontsize=12, transform=axes[1].transAxes)
    axes[2].text(x=0.01, y=1.06, s=f"(c)", fontsize=12, transform=axes[2].transAxes)

    # plt.legend(handles=legend_elements, bbox_to_anchor=(0.78, 0.98), loc='upper left')
    fig.subplots_adjust(wspace=0.1)


if __name__ == "__main__":
    plot_nonanticipative_matrix()
    plt.show()