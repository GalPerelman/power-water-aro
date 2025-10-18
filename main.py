import os
import time

import pandas as pd
import numpy as np
import argparse
import pickle
import yaml

import utils
from opt import RobustModel

np.set_printoptions(linewidth=10 ** 5)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.6f}'.format


def run_experiment(cfg):
    model = RobustModel(**cfg, omega=None)
    model.formulate_vectorized_opt_problem()
    df = pd.DataFrame()
    for _ in cfg['omegas']:
        model.omega = _
        t_start = time.time()
        model.solve()
        t_end = time.time()
        wc_cost, avg_cost, max_cost, csr = model.analyze_solution(n=1000)
        solution = {"pds_path": model.pds_path, "wds_path": model.wds_path,
                    "z0": model.z0_val, "z1": model.z1_val, "solver_obj": model.obj.value, "omega": _, "t": model.t,
                    "opt_method": model.opt_method, "pw_segments": 4, "elimination_method": model.elimination_method,
                    "n_bat_vars": model.n_bat_vars, 'manual_indep_variables': model.manual_indep_variables,
                    "pds_lags": model.pds_lags, "wds_lags": model.wds_lags}
        stats = {**{"total_time": t_end - t_start}, **model.solution_metrics}
        solution = {**solution, **stats}

        results = pd.DataFrame({"wc": wc_cost, "avg": avg_cost, "max": max_cost, "r": csr, **stats}, index=[len(df)])
        df = pd.concat([df, results])
        print(df)

        if cfg['export'] is not None:
            file_name = f"{cfg['name']}_{cfg['opt_method']}_{_}.pkl"
            export_path = os.path.join(cfg['export'], file_name)
            with open(export_path, 'wb') as handle:
                pickle.dump(solution, handle, protocol=pickle.HIGHEST_PROTOCOL)
                df.to_csv(os.path.join(cfg['export'], f"{cfg['name']}_{cfg['opt_method']}_summary.csv"))


def analyze_data_latency(cfg, omega=1, pds_latencies=None, wds_latencies=None):
    pds_latencies = [0, 2, 4, 6, 8, 10, 12] if pds_latencies is None else pds_latencies
    wds_latencies = [0, 2, 4, 6, 8, 10, 12] if wds_latencies is None else wds_latencies

    df = pd.DataFrame()
    for pds_lat in pds_latencies:
        for wds_lat in wds_latencies:
            model = RobustModel(**cfg, omega=None)
            model.pds_lat = pds_lat
            model.wds_lat = wds_lat
            model.formulate_vectorized_opt_problem()
            model.omega = omega
            model.solve()
            solution = {"pds_path": model.pds_path, "wds_path": model.wds_path, "z0": model.z0_val, "z1": model.z1_val,
                        "solver_obj": model.obj.value, "omega": omega, "t": model.t,"opt_method": model.opt_method,
                        "pw_segments": 4, "elimination_method": model.elimination_method,
                        "n_bat_vars": model.n_bat_vars, 'manual_indep_variables': model.manual_indep_variables}
            wc_cost, avg_cost, max_cost, csr = model.analyze_solution(n=1000)
            df = pd.concat([df, pd.DataFrame({"wc": wc_cost, "avg": avg_cost, "max": max_cost, "r": csr,
                                              "pds_lat": pds_lat, "wds_lat": wds_lat}, index=[len(df)])])
            print(df)
            if cfg['export'] is not None:
                file_name = f"{cfg['name']}_{cfg['opt_method']}_{omega}_pdslat-{pds_lat}_wdslat-{wds_lat}.pkl"
                export_path = os.path.join(cfg['export'], file_name)
                with open(export_path, 'wb') as handle:
                    pickle.dump(solution, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--command', type=str, required=True, choices=["experiment", "latency"])
    parser.add_argument('--omega', type=utils.int_or_float, default=1, required=False)
    parser.add_argument('--pds_lat', type=int, required=False, nargs='+')
    parser.add_argument('--wds_lat', type=int, required=False, nargs='+')

    args = parser.parse_args()

    if args.command == "experiment":
        with open(args.config_path) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            run_experiment(cfg)

    elif args.command == "latency":
        with open(args.config_path) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            analyze_data_latency(cfg, omega=args.omega, pds_latencies=args.pds_lat, wds_latencies=args.wds_lat)

