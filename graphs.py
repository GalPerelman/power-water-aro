import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker as mtick
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, AutoLocator, Locator
from matplotlib.lines import Line2D

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

    def tanks_volume(self, fig=None, color='C0', leg_label=""):
        ncols = max(1, int(math.ceil(math.sqrt(self.wds.n_tanks))))
        nrows = max(1, int(math.ceil(self.wds.n_tanks / ncols)))

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=self.wds.n_tanks, sharex=True, sharey=False, figsize=(9, 4.5))
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for tank_idx, (tank_name, tank_data) in enumerate(self.wds.tanks.iterrows()):
            inflow = self.x[:, self.model.n_pds + self.wds.n_combs + self.wds.n_desal + tank_idx] * self.wds.flows_factor
            if inflow.shape[0] != 1:
                init_vol = np.tile(tank_data['init_vol'], inflow.shape[0]).reshape(-1, 1) * self.wds.flows_factor
            else:
                init_vol = tank_data['init_vol'] * self.wds.flows_factor

            y = np.hstack([init_vol, tank_data['init_vol'] * self.wds.flows_factor + inflow.cumsum(axis=-1)]).T
            axes[tank_idx].plot(y, color, alpha=self.alpha)
            # axes[tank_idx].scatter(0, tank_data['init_vol'] * self.wds.flows_factor, facecolor="none", edgecolor='r')
            axes[tank_idx].grid(True)
            axes[tank_idx].hlines(tank_data['min_vol'] * self.wds.flows_factor, 0, self.model.t, 'r')
            axes[tank_idx].hlines(tank_data['max_vol'] * self.wds.flows_factor, 0, self.model.t, 'r')
            axes[tank_idx].hlines(tank_data['init_vol'] * self.wds.flows_factor, 0, self.model.t, 'k', linestyle='--')
            if leg_label:
                axes[tank_idx].plot(y[:, 0], color, label=leg_label)
                axes[tank_idx].legend(framealpha=1)

        fig.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig.text(0.02, 0.55, f'Volume ($m^3$)', va='center', rotation='vertical')
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

    def plot_generators(self, shared_y=False, fig=None, color="C0", leg_label="", zo=1):
        if fig is None:
            ncols = max(1, math.ceil(math.sqrt(self.pds.n_generators)))
            nrows = max(1, int(math.ceil(self.pds.n_generators / ncols)))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=shared_y, figsize=(9, 4.5))
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for i, (gen_idx, gen_data) in enumerate(self.pds.generators.iterrows()):
            p = self.x[:, self.model.n_bus + i] * self.pds.pu_to_mw
            axes[i].plot(p.T, color, alpha=self.alpha, zorder=zo)
            axes[i].set_axisbelow(True)
            axes[i].grid(True, zorder=0)
            # axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # axes[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
            # axes[i].yaxis.set_major_locator(ConstantAwareLocator(tol=1e-10, fixed_interval=1, fixed_range=20))
            axes[i].yaxis.set_major_locator(ConstantAwareLocator(tol=0.1, width=3, step=1))
            axes[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if leg_label:
                axes[i].plot(p.T[:, 0], color, label=leg_label)
                axes[i].legend(framealpha=1)

        fig.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig.text(0.02, 0.55, f'Generation ({self.pds.input_power_units.upper()})', va='center',
                 rotation='vertical')

        fig.subplots_adjust(left=0.1, bottom=0.15, top=0.95, right=0.92, wspace=0.2)
        return fig


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


def analyze_latency(aro_path):
    from scipy.interpolate import griddata
    df = pd.read_csv(aro_path, index_col=0)
    df = df.loc[df['theta'] != 0]

    # just for review the results
    df['penalty_deviation'] = df['penalized_avg_obj'] / df['avg_cost']
    df['penalty_cost'] = df['penalized_avg_obj'] - df['avg_cost']

    fig, ax = plt.subplots()
    ax.plot(df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 0), "avg_cost"],
            df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 0), "reliability"],
                    marker='o', mfc="w", c=COLORS["ARO-AVG"], label="ARO No Latency")

    ax.plot(df.loc[(df['pds_lat'] == 6) & (df['wds_lat'] == 0), "avg_cost"],
            df.loc[(df['pds_lat'] == 6) & (df['wds_lat'] == 0), "reliability"],
            marker='o', mfc="w", c="#00B8F5", label="PDS Latency=6 | WDS Latency=0")

    ax.plot(df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 6), "avg_cost"],
            df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 6), "reliability"],
            marker='o', mfc="w", c="#DC4141", label="PDS Latency=0 | WDS Latency=6")

    ax.plot(df.loc[(df['pds_lat'] == 12) & (df['wds_lat'] == 0), "avg_cost"],
            df.loc[(df['pds_lat'] == 12) & (df['wds_lat'] == 0), "reliability"],
            marker='o', mfc="w", label="PDS Latency=12 | WDS Latency=0")

    ax.plot(df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 12), "avg_cost"],
            df.loc[(df['pds_lat'] == 0) & (df['wds_lat'] == 12), "reliability"],
            marker='o', mfc="w", label="PDS Latency=0 | WDS Latency=12")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Reliability (%)")
    ax.legend(loc="lower right")

    num_categories = len(np.unique(df['pds_lat']))
    cmap = plt.get_cmap('RdBu_r', num_categories)  # using a colormap with discrete colors
    fig, axes = plt.subplots(ncols=2)
    axes[0].scatter(df['avg_cost'], df["reliability"], c=df['wds_lat'], cmap=cmap, edgecolor='k', alpha=0.8)
    axes[1].scatter(df['avg_cost'], df["reliability"], c=df['pds_lat'], cmap=cmap, edgecolor='k', alpha=0.8)

    fig, axes = plt.subplots(ncols=3, subplot_kw=dict(projection='3d'))
    for i, _ in enumerate([1, 2, 3]):
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
    fig, axes = plt.subplots(ncols=2, subplot_kw=dict(projection='3d'))
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
        ax.tick_params(axis='z', pad=8)
        ax.azim = -130
        ax.elev = 15

    axes[0].set_zlabel('Cost ($)', rotation=90)
    axes[1].set_zlabel('Reliability ($)', rotation=90)
    axes[0].zaxis.labelpad = 10


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