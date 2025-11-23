import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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


if __name__ == "__main__":
    computational_analysis()

