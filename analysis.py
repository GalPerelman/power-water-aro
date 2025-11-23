import numpy as np
import pandas as pd
import cvxpy as cp
import math
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker as mtick

import utils, graphs
from simulation import Simulation


def collect_problem_stats(prob):
    """
    Collect model-level and canonical (solver-level) size/sparsity stats for a CVXPY problem.

    Returns
    -------
    stats : dict
        {
          "model": {...},                  # variables & scalar constraint counts (pre-canonical)
          "solvers": { ... },              # per-solver canonicalization footprints (A/G/dims/nnz)
          "recommended": {...}             # a compact "paper-ready" view from best available solver
        }
    """

    # -------- helpers --------
    def _dims_get(dims, key, default=None):
        if dims is None:
            return default
        if isinstance(dims, dict):
            return dims.get(key, default)
        val = getattr(dims, key, None)
        return default if val is None else val

    def _mat_stats(M):
        if M is None:
            return None
        m, n = M.shape
        if sparse.issparse(M):
            nnz = int(M.nnz)
        else:
            nnz = int(np.count_nonzero(M))
        dens = (nnz / (m * n)) * 100.0 if m and n else 0.0
        return {"rows": int(m), "cols": int(n), "nnz": nnz,
                "density_pct": dens, "sparsity_pct": 100.0 - dens}

    # -------- model-level (pre-canonical) --------
    from cvxpy.constraints import Zero, NonPos, SOC, ExpCone
    model = {
        "num_vars": int(sum(v.size for v in prob.variables())),
        "num_constraint_objects": int(len(prob.constraints)),
        "scalar_counts_by_type": {
            "EQ": 0, "INEQ": 0, "SOC": 0, "EXP": 0, "OTHER": 0
        }
    }

    for con in prob.constraints:
        if isinstance(con, Zero):
            model["scalar_counts_by_type"]["EQ"] += int(con.size)
        elif isinstance(con, NonPos):
            model["scalar_counts_by_type"]["INEQ"] += int(con.size)
        elif isinstance(con, SOC):
            model["scalar_counts_by_type"]["SOC"] += int(con.size)
        elif isinstance(con, ExpCone):
            model["scalar_counts_by_type"]["EXP"] += int(con.size)
        else:
            model["scalar_counts_by_type"]["OTHER"] += int(con.size)

    # Totals (scalar constraints)
    model["num_scalar_equalities"] = model["scalar_counts_by_type"]["EQ"]
    model["num_scalar_inequalities"] = (
            model["scalar_counts_by_type"]["INEQ"]
            + model["scalar_counts_by_type"]["SOC"]
            + model["scalar_counts_by_type"]["EXP"]
            + model["scalar_counts_by_type"]["OTHER"]  # fallback bucket
    )

    # -------- solver-level (canonical) --------
    solver_order = [cp.ECOS, cp.SCS, cp.MOSEK, cp.CLARABEL]
    solvers = {}
    for s in solver_order:
        name = getattr(s, "name", str(s))
        try:
            data, inv, _ = prob.get_problem_data(s)
            dims = data.get("dims", {})
            # canonical vectors
            b = data.get("b", None)  # equalities
            h = data.get("h", None)  # inequalities
            c = data.get("c", None)  # canonical vars length
            # canonical matrices
            A = data.get("A", None)  # equalities
            G = data.get("G", None)  # inequalities

            # cone composition (keys might be absent for some backends)
            l = int(_dims_get(dims, "l", 0))
            ql = list(_dims_get(dims, "q", [])) or []
            sl = list(_dims_get(dims, "s", [])) or []
            rl = list(_dims_get(dims, "r", [])) or []  # rotated SOC
            ep = int(_dims_get(dims, "ep", 0))  # exponential

            solvers[name] = {
                "ok": True,
                "error": None,
                "cone_dims": {
                    "l": l,
                    "q_list": [int(x) for x in ql],
                    "s_list": [int(x) for x in sl],
                    "r_list": [int(x) for x in rl],
                    "ep": ep,
                    "num_soc_cones": len(ql),
                    "num_rsoc_cones": len(rl),
                    "num_sdp_blocks": len(sl),
                    "total_soc_dim": int(sum(int(x) for x in ql) + sum(int(x) for x in rl)),
                },
                "canonical_counts": {
                    "m_eq": int(len(b) if b is not None else 0),
                    "m_in": int(len(h) if h is not None else 0),
                    "n_canonical_vars": int(len(c) if c is not None else 0),
                },
                "A_stats": _mat_stats(A),
                "G_stats": _mat_stats(G),
            }
        except Exception as e:
            solvers[name] = {"ok": False, "error": repr(e)}

    preferred = None
    for cand in ["ECOS", "SCS", "MOSEK", "CLARABEL"]:
        if cand in solvers and solvers[cand].get("ok"):
            preferred = cand
            break

    # If none succeeded, leave recommended empty
    recommended = {}
    if preferred:
        srec = solvers[preferred]
        dims = srec["cone_dims"]
        counts = srec["canonical_counts"]
        recommended = {
            "solver": preferred,
            "cone_summary": {
                "LP_l_rows": dims["l"],
                "SOC_cones": dims["num_soc_cones"],
                "SOC_sizes": dims["q_list"],
                "RSOC_cones": dims["num_rsoc_cones"],
                "SDP_blocks": dims["num_sdp_blocks"],
                "EXP_rows": dims["ep"],
            },
            "canonical_sizes": {
                "equalities_rows_m_eq": counts["m_eq"],
                "inequalities_rows_m_in": counts["m_in"],
                "canonical_variables_n": counts["n_canonical_vars"],
                "A": srec["A_stats"],
                "G": srec["G_stats"],
            }
        }

    return {"model": model, "solvers": solvers, "recommended": recommended}


def _dims_get(dims, key, default=None):
    """Robust getter for CVXPY dims (dict or ConeDims object)."""
    if dims is None:
        return default
    # dict-like
    if isinstance(dims, dict):
        return dims.get(key, default)
    # object-like (ConeDims)
    val = getattr(dims, key, None)
    return default if val is None else val


def get_cone_counts(prob):
    dims = None
    for solver in (cp.ECOS):  # solvers that expose dims
        try:
            data, _, _ = prob.get_problem_data(solver)
            if "dims" in data and data["dims"] is not None:
                dims = data.get("dims", {})
                break
        except Exception:
            continue
    if dims is None:
        return {"num_soc": None, "num_rsoc": None, "num_sdp": None}

    q = _dims_get(dims, "q", [])  # second-order cones
    r = _dims_get(dims, "r", [])  # rotated SOCs (present in some CVXPY/SCS versions)
    s = _dims_get(dims, "s", [])  # SDP blocks (likely empty here)

    # Normalize to Python lists of ints
    q = list(map(int, list(q))) if hasattr(q, "__iter__") else []
    r = list(map(int, list(r))) if hasattr(r, "__iter__") else []
    s = list(map(int, list(s))) if hasattr(s, "__iter__") else []

    return {
        "num_soc": len(q),
        "num_rsoc": len(r),
        "num_sdp": len(s),
        "total_soc_dim": sum(q) + sum(r)  # total second-order dimensions
    }


def analyze_problem_structure(prob):
    out = {}
    # High-level model sizes (pre-canonicalization)
    out["model_num_vars"] = int(sum(v.size for v in prob.variables()))

    # High level constraints
    constraint_types = {}
    constraint_sizes = {}
    constraint_details = []
    for con in prob.constraints:
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
                    if any(keyword in arg_type.lower() for keyword in
                           ['norm', 'soc', 'quadover', 'quad', 'pnorm']):
                        is_soc = True
                        break
                    # Also check string representation of args
                    if 'norm2' in str(arg).lower():
                        is_soc = True
                        break

        constraint_info['is_soc'] = is_soc
        constraint_details.append(constraint_info)
    out['constraints'] = constraint_details

    # --- Get a solver-level sparse matrix to measure size/sparsity (no solve) ---
    data_any = None
    for solver in (cp.MOSEK, cp.SCS, cp.ECOS, cp.CLARABEL, cp.SCS):
        try:
            data_any, _, _ = prob.get_problem_data(solver)
            break
        except Exception:
            continue

    A_mat = None
    if data_any:
        # pick the largest genuine 2-D constraint matrix; skip vectors like 'c', 'b'
        candidates = []
        for k, v in data_any.items():
            shp = getattr(v, "shape", None)
            if shp and len(shp) == 2 and shp[0] > 0 and shp[1] > 0:
                candidates.append((k, v, shp[0] * shp[1]))
        if candidates:
            # choose the biggest by element count
            candidates.sort(key=lambda kvs: kvs[2], reverse=True)
            A_key, A_mat, _ = candidates[0]
            out["solver_matrix_key"] = A_key  # often 'A' for SCS/ECOS

    if A_mat is not None:
        m, n = A_mat.shape
        out["solver_rows"] = int(m)
        out["solver_cols"] = int(n)
        nnz = int(A_mat.nnz) if sparse.issparse(A_mat) else int(np.count_nonzero(A_mat))
        out["nnz"] = nnz
        density = (nnz / (m * n)) if (m * n) else 0.0
        out["constraint_matrix_density_pct"] = 100.0 * density
        out["constraint_matrix_sparsity_pct"] = 100.0 * (1.0 - density)
    else:
        out.update({
            "solver_rows": None, "solver_cols": None, "nnz": None,
            "constraint_matrix_density_pct": None, "constraint_matrix_sparsity_pct": None
        })

    return out


def computational_analysis(thetas=[0.5, 1, 1.5, 2, 2.5, 3]):
    df = pd.DataFrame()
    for case in ['I_ro/I_3-bus-desalination-wds_ro',
                 'I_aro/I_3-bus-desalination-wds_aro',
                 'II_ro/II_ieee9-national-wds_ro',
                 'II_aro/II_ieee9-national-wds_aro',
                 'III_ro/III_ieee14-national-wds_ro',
                 'III_aro/III_ieee14-national-wds_aro',
                 'IV_ro/IV_ieee24-national-wds_ro',
                 'IV_aro/IV_ieee24-national-wds_aro'
                 ]:

        # do only once for each case
        try:
            solution = utils.read_solution(sol_path=f"output/{case}_{0.5}.pkl")
            sim = Simulation(**solution, plot=False)
            sim.formulate_vectorized_opt_problem()
            stats = analyze_problem_structure(sim.problem)
        except FileNotFoundError:
            continue

        for theta in thetas:
            try:
                solution = utils.read_solution(sol_path=f"output/{case}_{theta}.pkl")
            except FileNotFoundError:
                continue

            print(stats['constraints'])
            if len(stats['constraints']) == 3:
                # these are the constraints that blocks the non-anticipitavity matrix
                lin_equality_constraints = stats['constraints'][1]['size']
            else:
                lin_equality_constraints = 0

            temp = pd.DataFrame({
                'case': case.split('/')[0],
                'method': (case.split('/')[0]).split('_')[1],
                'model_num_vars': stats['model_num_vars'],
                'soc_constraints': stats['constraints'][0]['size'],  # these are the problem physical constraints
                'lin_equality_constraints': lin_equality_constraints,
                'solver_rows': stats['solver_rows'],
                'solver_cols': stats['solver_cols'],
                'nnz': stats['nnz'],
                'constraint_matrix_density_pct': stats['constraint_matrix_density_pct'],

                'compilation_time': solution['compilation_time'],
                'solver_time': solution['solver_time'],
                'status': solution['status'],
                'pds_lags': solution['pds_lags'], 'wds_lags': solution['wds_lags']

            }, index=[len(df)])
            df = pd.concat([df, temp])

    print(df)
    agg_dict = {col: 'mean' for col in df.select_dtypes(include='number').columns}
    agg_dict['compilation_time'] = ['mean', 'std']
    agg_dict['solver_time'] = ['mean', 'std']
    result = df.groupby(['case', 'method']).agg(agg_dict)
    print(result)
    df.to_csv("full_computational_analysis.csv")
    result.to_csv("aggregated_computational_analysis.csv")


def plot_ro_vs_aro(ro_path, aro_path, thetas, n, method_in_front="aro", shared_y=True, case_name="I"):
    batteries_fig, batteries_ax = plt.subplots()

    if method_in_front == "aro":
        z_ro, z_aro = 5, 10
    else:
        z_ro, z_aro = 10, 5

    if case_name == "III":
        skip_axes = 2
    else:
        skip_axes = None

    for i, theta in enumerate(thetas):
        ro = utils.read_solution(sol_path=f"{ro_path}_{theta}.pkl")
        sim = Simulation(**ro, plot=False, n=n)
        costs, violations_rate = sim.run()
        tanks_fig = sim.graphs.tanks_volume(color=graphs.COLORS["RO-MAX"], shared_y=shared_y)
        gen_fig = sim.graphs.plot_generators(shared_y=shared_y, color=graphs.COLORS["RO-MAX"], leg_label="", zo=z_ro,
                                             skip_axes=skip_axes)
        for bat_idx, (bat_name, bat_data) in enumerate(sim.pds.batteries.iterrows()):
            batteries_ax.hlines(bat_data['min_storage'], 0, sim.t, 'k', linewidth=1.5, zorder=50)
            batteries_ax.hlines(bat_data['max_storage'], 0, sim.t, 'k', linewidth=1.5, zorder=50,
                                label="Battery Limits")
            batteries_ax.hlines(bat_data['init_storage'], 0, sim.t, 'k', linestyle='--', zorder=50,
                                label="Initial SOC")
            batteries_ax.plot(sim.solution["batteries"][bat_name].T, graphs.COLORS["RO-MAX"], alpha=0.3, zorder=z_ro)
            batteries_ax.grid(True)
            batteries_ax.plot(sim.solution["batteries"][bat_name].T[:, 0], graphs.COLORS["RO-MAX"],
                              label="RO")  # for legend

        aro = utils.read_solution(sol_path=f"{aro_path}_{theta}.pkl")
        sim = Simulation(**aro, plot=False, n=n)
        costs, violations_rate = sim.run()

        tanks_fig = sim.graphs.tanks_volume(fig=tanks_fig, color="C0", shared_y=shared_y)
        tanks_custom_handles = [
            mlines.Line2D([0], [0], color='k', label='Tank Limits'),
            mlines.Line2D([0], [0], color='k', label='Initial Volume', linestyle='--'),
            mlines.Line2D([0], [0], color=graphs.COLORS["RO-MAX"], label='RO'),
            mlines.Line2D([0], [0], color="C0", label='ARO'),
        ]

        gen_fig = sim.graphs.plot_generators(shared_y=shared_y, fig=gen_fig, color="C0", leg_label="", zo=z_aro,
                                             skip_axes=skip_axes)
        gen_custom_handles = [
            mlines.Line2D([0], [0], color=graphs.COLORS["RO-MAX"], label='RO'),
            mlines.Line2D([0], [0], color="C0", label='ARO'),
        ]

        if case_name == "I":
            tanks_fig.legend(handles=tanks_custom_handles, loc='upper left', bbox_to_anchor=(0.12, 1.0),
                             ncol=4, columnspacing=0.8)
            tanks_fig.subplots_adjust(left=0.14, right=0.95, wspace=0.14, top=0.83)
            tanks_fig.text(0.55, 0.04, 'Time (hr)', ha='center')
            tanks_fig.text(0.02, 0.55, f'Volume ($m^3$)', va='center', rotation='vertical')
            tanks_fig.set_size_inches(7, 4.2)

            gen_fig.legend(handles=gen_custom_handles, loc='upper right', bbox_to_anchor=(0.94, 0.92))
            gen_fig.subplots_adjust(right=0.8, wspace=0.15, top=0.9)
            gen_fig.text(0.45, 0.04, 'Time (hr)', ha='center')
            gen_fig.text(0.02, 0.55, f'Generation ({sim.pds.input_power_units.upper()})', va='center',
                         rotation='vertical')
            gen_fig.set_size_inches(8, 4)

        if case_name == "III":
            tanks_fig.legend(handles=tanks_custom_handles, loc='upper right', bbox_to_anchor=(0.9, 0.5))
            tanks_fig.subplots_adjust(right=0.97, wspace=0.3, top=0.92, bottom=0.12, hspace=0.23)
            tanks_fig.text(0.5, 0.02, 'Time (hr)', ha='center')
            tanks_fig.text(0.02, 0.55, f'Volume ($m^3$)', va='center', rotation='vertical')
            tanks_fig.set_size_inches(9, 4)

            gen_fig.legend(handles=gen_custom_handles, loc='upper right', bbox_to_anchor=(0.84, 0.94))
            gen_fig.subplots_adjust(right=0.97, wspace=0.3, top=0.92, bottom=0.12, hspace=0.23)
            gen_fig.text(0.5, 0.02, 'Time (hr)', ha='center')
            gen_fig.text(0.02, 0.55, f'Generation ({sim.pds.input_power_units.upper()})', va='center',
                         rotation='vertical')
            gen_fig.set_size_inches(8, 4)

        for bat_idx, (bat_name, bat_data) in enumerate(sim.pds.batteries.iterrows()):
            batteries_ax.plot(sim.solution["batteries"][bat_name].T, 'C0', alpha=0.3, zorder=z_aro)
            batteries_ax.grid(True)
            batteries_ax.plot(sim.solution["batteries"][bat_name].T[:, 0], 'C0', label="ARO")  # for legend

        leg = batteries_ax.legend(framealpha=1, loc="upper left", bbox_to_anchor=(-0.018, 1.12),
                                  ncol=4, columnspacing=0.8)
        leg.set_zorder(50)
        batteries_ax.set_xlabel('Time (hr)')
        batteries_ax.set_ylabel(f'SOC ({sim.pds.input_power_units.upper()}h)')


def analyze_adaptability(
    ro_path,
    aro_path,
    scenarios_path,
    scenario_names,
    parameters_to_plot,
    *,  # All arguments after this must be passed by name (keyword arguments), not by position.

    # ---- Selection controls ----
    include_generators="all",      # "all" | list of generator indices (as in sim.pds.generators index)
    include_tanks="all",           # "all" | list of tank names (keys of sim.solution["tanks"])
    include_batteries="all",       # "all" | list of battery names (keys of sim.solution["batteries"])
    include_stations="all",        # "all" | list of station names (sim.wds.combs['station'].unique())
    include_desal="all",           # "all" | list of desal names (sim.wds.desal.index or .name)
    include_params="auto",         # "auto" (=parameters_to_plot) | explicit list like ["load", "pv", "demand"]

    # ---- Layout controls ----
    ncols=None,                    # None => auto sqrt; otherwise fixed number of columns
    sharex=True,
    figsize=(14, 8),
    group_rows=True,  # if True, each type gets its own row block (params / gens / tanks / bats / stations / desal)
    type_order=("params", "generators", "batteries", "stations", "desal", "tanks"),
    # ---- Styling / legend ----
    legend=True,
    legend_fontsize=10,
    legend_loc="center left",
    legend_ncols=1,
    # ---- RO overlay controls ----
    show_ro_for=("generators", "tanks", "batteries", "stations", "desal", "params"),  # any subset of types
    ro_line_kwargs=None,           # dict for RO line style, e.g. {"color":"k","lw":1.8,"alpha":0.9}
    scenario_line_kwargs=None,     # base kwargs for scenario lines (applied in addition to label)
):
    """
    Flexible, grid-based adaptability plotter for ARO vs. scenarios with optional RO overlays.

    Selection arguments can be "all" or explicit lists (indices for generators; names for others).
    Set group_rows=True to assign one row per type block in `type_order`.
    """

    # ---- Helpers ----
    def _normalize_selection(found, selection):
        if selection == "all":
            return list(found)
        # Keep only items that actually exist
        return [x for x in selection if x in found]

    ro_line_kwargs = ro_line_kwargs or {"color": "k", "zorder": 1}
    scenario_line_kwargs = scenario_line_kwargs or {}

    # ---- Read ARO + scenarios and run once per scenario ----
    aro_solution = utils.read_solution(sol_path=f"{aro_path}")
    scenarios = pd.read_csv(scenarios_path)

    # Dry run for sizing/metadata
    sim0 = Simulation(**aro_solution, plot=False, n=1)
    formatter = mtick.ScalarFormatter(useOffset=False, useMathText=False)
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, _: f"{int(x)}")

    # Parameter selection
    params = parameters_to_plot if include_params == "auto" else include_params
    params = list(params or [])

    # Discover available items
    gen_indices = list(sim0.pds.generators.index)
    tank_names = list(sim0.wds.tanks.index)
    bat_names = list(sim0.pds.batteries.index)
    station_names = list(sim0.wds.combs['station'].unique())
    desal_names = list(sim0.wds.desal.index)

    # Apply selections
    gen_indices = _normalize_selection(gen_indices, include_generators)
    tank_names = _normalize_selection(tank_names, include_tanks)
    bat_names = _normalize_selection(bat_names, include_batteries)
    station_names = _normalize_selection(station_names, include_stations)
    desal_names = _normalize_selection(desal_names, include_desal)

    # Count plots by type
    counts = {
        "params": len(params),
        "generators": len(gen_indices),
        "tanks": len(tank_names),
        "batteries": len(bat_names),
        "stations": len(station_names),
        "desal": len(desal_names),
    }
    total_plots = sum(counts[t] for t in type_order)

    # ---- Layout calculation ----
    if total_plots == 0:
        raise ValueError("No plots requested. Check your include_* selections and parameters_to_plot.")

    if ncols is None:
        ncols_auto = max(1, int(math.ceil(math.sqrt(total_plots))))
    else:
        ncols_auto = int(ncols)

    if group_rows:
        # One row-block per type that has >0 plots; cols = min(ncols_auto, max count in any active type)
        active_types = [t for t in type_order if counts[t] > 0]
        # per-type columns = min(ncols_auto, count) but grid uses a uniform ncols across all rows
        ncols_eff = max(1, ncols_auto)
        nrows_eff = sum(int(math.ceil(counts[t] / ncols_eff)) if counts[t] > 0 else 0 for t in active_types) - 3
    else:
        ncols_eff = max(1, ncols_auto)
        nrows_eff = max(1, int(math.ceil(total_plots / ncols_eff))) - 3

    # Reserve one extra cell for legend if possible
    total_cells = nrows_eff * ncols_eff

    fig, axes = plt.subplots(nrows=nrows_eff, ncols=ncols_eff, sharex=sharex, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    # Map each plot slot to an axis in a deterministic order
    ax_map = {"params": [], "generators": [], "batteries": [], "stations": [], "desal": [], "tanks": []}

    def _assign_axes_grouped():
        idx = 0
        for t in type_order:
            k = counts[t]
            if k == 0:
                continue
            rows_needed = int(math.ceil(k / ncols_eff))
            # fill row by row for this type
            for _ in range(k):
                ax_map[t].append(axes[idx])
                idx += 1

    def _assign_axes_flat():
        idx = 0
        for t in type_order:
            for _ in range(counts[t]):
                ax_map[t].append(axes[idx])
                idx += 1

    if group_rows:
        _assign_axes_grouped()
    else:
        _assign_axes_flat()

    # Convenience to format y-axis
    def _style_axis(ax, title, ylabel):
        ax.set_axisbelow(True)
        ax.grid(True, zorder=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(graphs.ConstantAwareLocator(tol=0.5, width=3, step=1))
        ax.yaxis.set_major_formatter(formatter)

    # ---- Plot ARO across scenarios ----
    for s_name in scenario_names:
        scenario_values = scenarios[s_name].values.reshape(-1, 1)
        sim = Simulation(**aro_solution, plot=False, n=1, sample=scenario_values)
        sim.run()  # fills fields used below

        # Params: "load", "pv", "demand"
        for i, param in enumerate(params):
            ax = ax_map["params"][i]
            if param.startswith("load"):
                load_sample = sim.sample_loads.reshape(-1, sim.n_bus, 1) * sim.pds.pu_to_mw
                ax.plot(load_sample.sum(axis=1), label=s_name, **scenario_line_kwargs[s_name])
                _style_axis(ax, "Total Load", f'Power ({sim.pds.input_power_units.upper()})')

            elif param.startswith("pv"):
                pv_node_idx = sim.pds.bus[sim.pds.bus['max_pv_pu'] > 0].index[0]
                pv_sample = sim.sample_pv.reshape(-1, sim.n_bus, 1) * sim.pds.pu_to_mw
                ax.plot(pv_sample[:, pv_node_idx, :], label=s_name, **scenario_line_kwargs[s_name])
                _style_axis(ax, "PV Injection", f'Power ({sim.pds.input_power_units.upper()})')

            elif param.startswith("demand"):
                dem_sample = sim.sample_dem.reshape(-1, sim.n_tanks, sim.n) * sim.wds.flows_factor
                ax.plot(dem_sample.sum(axis=1), label=s_name, **scenario_line_kwargs[s_name])
                _style_axis(ax, "Total Demand", r'Demand ($m^3/hr$)')

        # Generators
        for j, gen_idx in enumerate(gen_indices):
            ax = ax_map["generators"][j]
            # map gen_idx -> column order (j) as in x_by_sample: your original used position i
            # We assume sim.pds.generators is aligned with indices; find positional index:
            pos = list(sim.pds.generators.index).index(gen_idx)
            p = sim.x_by_sample[:, sim.n_bus + pos] * sim.pds.pu_to_mw
            ax.plot(p.T, label=s_name, **scenario_line_kwargs[s_name])
            _style_axis(ax, f'Generator {gen_idx+1}', f'Power ({sim.pds.input_power_units.upper()})')

        # Batteries
        for j, bat_name in enumerate(bat_names):
            ax = ax_map["batteries"][j]
            ax.plot(sim.solution["batteries"][bat_name][0], label=s_name, **scenario_line_kwargs[s_name])
            _style_axis(ax, f'Battery {bat_name}', f'SOC ({sim.pds.input_power_units.upper()}h)')

        # Stations (sum flows by station)
        # Build indices for combs once
        combs_idx = [sim.n_pds + _ for _ in range(sim.n_combs)]
        combs_flow = sim.wds.combs['flow'].values.reshape(1, len(sim.wds.combs['flow'].values), 1)
        pumps_flow = sim.x_by_sample[:, combs_idx, :] * combs_flow * sim.wds.flows_factor

        # Pumping Stations
        for j, station_name in enumerate(station_names):
            ax = ax_map["stations"][j]
            station_idx = sim.wds.combs.index[sim.wds.combs['station'] == station_name]
            ax.plot(pumps_flow[:, station_idx, :].sum(axis=1).T, label=s_name, **scenario_line_kwargs[s_name])
            _style_axis(ax, f'Pumping Station {station_name}', r'Flow ($m^3/hr$)')

        # Desal
        desal_base_idx = [sim.n_pds + sim.n_combs + k for k in range(sim.n_desal)]
        desal_flow = sim.x_by_sample[:, desal_base_idx, :] * sim.wds.flows_factor
        desal_all_names = list(sim.wds.desal.index)
        for j, dname in enumerate(desal_names):
            ax = ax_map["desal"][j]
            pos = desal_all_names.index(dname)
            ax.plot(desal_flow[:, pos, :][0], label=s_name, **scenario_line_kwargs[s_name])
            _style_axis(ax, f'Desalination {dname + 1}', r'Flow ($m^3/hr$)')

        # Tanks
        for j, tank_name in enumerate(tank_names):
            ax = ax_map["tanks"][j]
            tank_row = sim.wds.tanks.loc[tank_name]
            init_vol = np.array([tank_row['init_vol'] * sim.wds.flows_factor]).reshape(-1, 1)
            # find by name within solution dict
            y = np.hstack([init_vol, sim.solution["tanks"][tank_name][0].reshape(-1, 1).T * sim.wds.flows_factor])[
                0]
            ax.plot(y, label=s_name, **scenario_line_kwargs[s_name])
            _style_axis(ax, f'Tank {tank_name}', r'Volume ($m^3$)')

    # ---- RO overlays where requested ----
    if show_ro_for:
        ro_solution = utils.read_solution(sol_path=f"{ro_path}")
        sim_ro = Simulation(**ro_solution, plot=False, n=1)
        sim_ro.run()

        # Params overlay (if available)
        if "params" in show_ro_for:
            for i, param in enumerate(params):
                ax = ax_map["params"][i]
                if param.startswith("load"):
                    load_sample = sim_ro.sample_loads.reshape(-1, sim_ro.n_bus, 1) * sim_ro.pds.pu_to_mw
                    ax.plot(load_sample.sum(axis=1), label="RO", **ro_line_kwargs)
                elif param.startswith("pv"):
                    pv_node_idx = sim_ro.pds.bus[sim_ro.pds.bus['max_pv_pu'] > 0].index[0]
                    pv_sample = sim_ro.sample_pv.reshape(-1, sim_ro.n_bus, 1) * sim_ro.pds.pu_to_mw
                    ax.plot(pv_sample[:, pv_node_idx, :], label="RO", **ro_line_kwargs)
                elif param.startswith("demand"):
                    dem_sample = sim_ro.sample_dem.reshape(-1, sim_ro.n_tanks, sim_ro.n) * sim_ro.wds.flows_factor
                    ax.plot(dem_sample.sum(axis=1), label="RO", **ro_line_kwargs)

        # Generators
        if "generators" in show_ro_for and gen_indices:
            gen_all_idx = list(sim_ro.pds.generators.index)
            for j, gen_idx in enumerate(gen_indices):
                ax = ax_map["generators"][j]
                pos = gen_all_idx.index(gen_idx)
                p = sim_ro.x_by_sample[:, sim_ro.n_bus + pos] * sim_ro.pds.pu_to_mw
                ax.plot(p.T, label="RO", **ro_line_kwargs)

        # Batteries
        if "batteries" in show_ro_for and bat_names:
            for j, bat_name in enumerate(bat_names):
                ax = ax_map["batteries"][j]
                ax.plot(sim_ro.solution["batteries"][bat_name][0], label="RO", **ro_line_kwargs)

        # Pumping Stations
        if "stations" in show_ro_for and station_names:
            combs_idx = [sim_ro.n_pds + _ for _ in range(sim_ro.n_combs)]
            combs_flow = sim_ro.wds.combs['flow'].values.reshape(1, len(sim_ro.wds.combs['flow'].values), 1)
            pumps_flow = sim_ro.x_by_sample[:, combs_idx, :] * combs_flow * sim_ro.wds.flows_factor
            for j, station_name in enumerate(station_names):
                ax = ax_map["stations"][j]
                station_idx = sim_ro.wds.combs.index[sim_ro.wds.combs['station'] == station_name]
                ax.plot(pumps_flow[:, station_idx, :].sum(axis=1).T, label="RO", **ro_line_kwargs)

        # Desal
        if "desal" in show_ro_for and desal_names:
            desal_base_idx = [sim_ro.n_pds + sim_ro.n_combs + k for k in range(sim_ro.n_desal)]
            desal_flow = sim_ro.x_by_sample[:, desal_base_idx, :] * sim_ro.wds.flows_factor
            desal_all_names = list(sim_ro.wds.desal.index)
            for j, dname in enumerate(desal_names):
                ax = ax_map["desal"][j]
                pos = desal_all_names.index(dname)
                ax.plot(desal_flow[:, pos, :][0], label="RO", **ro_line_kwargs)

        # Tanks
        if "tanks" in show_ro_for and tank_names:
            for j, tank_name in enumerate(tank_names):
                ax = ax_map["tanks"][j]
                tank_row = sim_ro.wds.tanks.loc[tank_name]
                init_vol = np.array([tank_row['init_vol'] * sim_ro.wds.flows_factor]).reshape(-1, 1)
                y = np.hstack([init_vol, sim_ro.solution["tanks"][tank_name][0].reshape(-1, 1).T
                               * sim_ro.wds.flows_factor])[0]
                ax.plot(y, label="RO", **ro_line_kwargs)

    # ---- Legend handling ----
    if legend:
        # 1) Collect handles/labels from ALL plotted axes (not just the first one)
        used_axes = [ax for lst in ax_map.values() for ax in lst]

        handles_by_label = {}  # keep first handle per label (preserves RO style)
        for ax in used_axes:
            h, l = ax.get_legend_handles_labels()
            l = ["ARO - " + lab if lab != "RO" else lab for lab in l]  # prefix ARO to scenario labels
            for hh, ll in zip(h, l):
                if ll and ll not in handles_by_label:
                    handles_by_label[ll] = hh

        if handles_by_label:
            # 2) Order: scenarios in given order, then RO, then any extras
            ordered_labels = [s for s in scenario_names if s in handles_by_label]
            if "RO" in handles_by_label:
                ordered_labels.append("RO")
            # add any leftovers (unlikely, but just in case)
            ordered_labels += [lab for lab in handles_by_label.keys() if lab not in ordered_labels]

            handles = [handles_by_label[lab] for lab in ordered_labels]
            labels = ordered_labels

            # 3) Place legend in first free axis; fallback to slim right-side axis
            all_axes = list(axes)
            free_axes = [ax for ax in all_axes if ax not in used_axes]
            if free_axes:
                legend_ax = free_axes[0]
                legend_ax.axis('off')
                legend_ax.legend(handles, labels, loc=legend_loc, fontsize=legend_fontsize, ncols=legend_ncols)
            else:
                fig.subplots_adjust(right=0.86)
                legend_ax = fig.add_axes([0.88, 0.15, 0.10, 0.70])
                legend_ax.axis('off')
                legend_ax.legend(handles, labels, loc="center", fontsize=legend_fontsize, ncols=legend_ncols)

    # Turn off any axes not used by plots or the legend
    used_axes = [ax for lst in ax_map.values() for ax in lst]
    if 'legend_ax' in locals() and legend_ax is not None:
        used_axes.append(legend_ax)

    for ax in axes:
        if ax not in used_axes:
            ax.axis('off')

    fig.align_ylabels()

    # --- Ensure bottom-row subplots show xtick labels ---
    for ax in axes:
        if ax.get_visible() and ax.get_frame_on():
            ax.tick_params(labelbottom=True)

    fig.text(0.5, 0.02, 'Time (hr)', ha='center')
    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.08, top=0.96, wspace=0.38, hspace=0.6)
    return fig, ax_map


if __name__ == "__main__":
    computational_analysis()

    ##############################   plot operational results   ##############################
    plot_ro_vs_aro("output/I_ro/I_3-bus-desalination-wds_ro",
                   aro_path="output/I_aro/I_3-bus-desalination-wds_aro", thetas=[1], n=500)  # n=100 for generators
    plot_ro_vs_aro("output/I_ro_sa/I_sa_dep_bat_ro", aro_path="output/I_aro_sa/I_sa_dep_bat_aro",
                   thetas=[1], n=500, method_in_front="aro")  # n=100 for generators
    plot_ro_vs_aro("output/III_ro/III_ieee14-national-wds_ro",
                    aro_path="output/III_aro/III_ieee14-national-wds_aro",
                    thetas=[1], n=500, shared_y=False, case_name="III")

    ##############################   latency analysis plot   ##############################
    graphs.analyze_latency(aro_path="output/III_aro/III_ieee14-national-wds_aro_por.csv", thetas=[1])


    ##############################   adaptability analysis plot   ##############################
    fig, ax_map = analyze_adaptability(
        ro_path="output/III_ro/III_ieee14-national-wds_ro_1.pkl",
        aro_path="output/III_aro/III_ieee14-national-wds_aro_1.pkl",
        scenarios_path="output/III_aro/III_aro_adaptability_scenarios.csv",
        scenario_names=["High Load Start at 12", "High Demand", "Nominal"],  # order will determine the zorder, last is on top
        parameters_to_plot=["load", "demand"],
        include_generators="all",
        include_tanks="all",
        include_batteries=[],
        include_stations="all",
        include_desal="all",
        group_rows=True,
        show_ro_for=(),
        scenario_line_kwargs={"High Load Start at 12": {"color": "C1", "lw": 3, "alpha": 0.8},
                              "High Demand": {"color": "C0", "lw": 3, "alpha": 0.8},#053e7a, #f1be04
                              "Nominal": {"color": "k", "lw": 0.8}},
        ncols=4,
        legend_ncols=1,
        legend_loc="upper left"
    )
