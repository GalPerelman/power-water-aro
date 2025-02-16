import os

import pandas as pd
import numpy as np
import yaml

import utils


class PDS:
    def __init__(self, data_folder: str):
        """
        The units of the input data files need to be consistent
        Basic units for power need to be defined in the params.yaml file to be one of:
        w, kw, or mw

        file            parameter           units
        ==========================================
        bus             Vmin_pu             pu
        bus             Vmax_pu             pu
        lines           r_ohm               ohm
        lines           x_ohm               ohm
        lines           r_pu                pu
        lines           x_pu                pu

        active_dem:     all                 power units
        reactive_dem:   all                 power units
        tariffs:        all                 S/power units

        generators      min_gen_p           power units
        generators      max_gen_p           power units
        generators      min_gen_q           power units
        generators      max_gen_q           power units
        generators      a                   $/(power units^2) * h
        generators      b                   $/power units * h
        generators      c                   $

        batteries       min_storage         power units * h
        batteries       max_storage         power units * h
        batteries       max_charge          power units
        batteries       init_storage        power units * h
        batteries       final_storage       power units * h

        """
        self.data_folder = data_folder

        # default parameters
        self.input_power_units = "mw"  # mw is default, other units to be defined in params.yaml
        self.nominal_voltage_kv = 100
        self.power_base_mva = 100
        self.active_demand_factor = 1
        self.reactive_demand_factor = 1

        # read and update defaults parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

        self.bus = pd.read_csv(os.path.join(self.data_folder, 'bus.csv'), index_col=0)
        self.lines = pd.read_csv(os.path.join(self.data_folder, 'lines.csv'), index_col=0)
        self.dem_active = pd.read_csv(os.path.join(self.data_folder, 'dem_active_power.csv'), index_col=0)
        self.dem_reactive = pd.read_csv(os.path.join(self.data_folder, 'dem_reactive_power.csv'), index_col=0)
        self.tariffs = pd.read_csv(os.path.join(self.data_folder, 'tariffs.csv'), index_col=0)
        self.generators = pd.read_csv(os.path.join(self.data_folder, 'generators.csv'), index_col=0)
        self.pv = pd.read_csv(os.path.join(self.data_folder, 'pv.csv'), index_col=0)
        self.batteries = pd.read_csv(os.path.join(self.data_folder, 'batteries.csv'), index_col=0)
        self.max_gen_profile = pd.read_csv(os.path.join(self.data_folder, 'max_gen_profile.csv'), index_col=0)
        self.pumps_bus = pd.read_csv(os.path.join(self.data_folder, 'pumps_bus.csv'), index_col=0).fillna(0)
        self.desal_bus = pd.read_csv(os.path.join(self.data_folder, 'desal_bus.csv'), index_col=0).fillna(0)

        self.dem_active_std = pd.read_csv(os.path.join(self.data_folder, 'dem_active_power_std.csv'), index_col=0)
        self.pv_std = pd.read_csv(os.path.join(self.data_folder, 'pv_std.csv'), index_col=0)


        try:
            # optional input
            self.bus_criticality = pd.read_csv(os.path.join(self.data_folder, 'criticality.csv'), index_col=0)
        except FileNotFoundError:
            self.bus_criticality = None

        self.n_bus = len(self.bus)
        self.n_lines = len(self.lines)
        self.n_generators = len(self.generators)
        self.n_pv = len(self.pv)
        self.n_batteries = len(self.batteries)
        self.n_loads = len(self.dem_active.loc[self.dem_active.sum(axis=1) > 0].index.to_list())

        self.factorize_demands()
        self.gen_mat = utils.get_mat_for_type(self.bus, self.generators)
        self.bat_mat = utils.get_mat_for_type(self.bus, self.batteries)
        self.construct_generators_params()
        self.construct_pv_params()
        self.construct_batteries_params()

        # unit conversion in the end of the initiation
        self.to_pu, self.pu_to_power_input_units, self.pu_to_kv = self.convert_to_pu()
        self.pu_to_kw, self.pu_to_mw = self.power_base_mva * 10 ** 6 / 1000, self.power_base_mva

        self.y = self.get_admittance_mat()

    def factorize_demands(self):
        try:
            self.dem_active *= self.active_demand_factor
        except AttributeError:
            pass

        try:
            self.dem_reactive *= self.reactive_demand_factor
        except AttributeError:
            pass

    def convert_to_pu(self):
        """
        Function for converting input data to PU (dimensionless per unit)
        The conversion is done by remove all magnitude units (Kilo, Mega etc.) and divide by base values
        base voltage and base power - defined in the input params.yaml file
        return:
            values to inverse conversion back to physical units
            pu_to_kw, pu_to_kv
        """
        c = utils.POWER_UNITS[self.input_power_units]  # coefficient to convert power units to watt

        z = ((self.nominal_voltage_kv * 1000) ** 2) / (self.power_base_mva * 10 ** 6)
        if 'r_ohm' in self.lines.columns:
            self.lines['r_pu'] = self.lines['r_ohm'] / z
        if 'x_ohm' in self.lines.columns:
            self.lines['x_pu'] = self.lines['x_ohm'] / z

        self.dem_active = (self.dem_active * c) / (self.power_base_mva * 10 ** 6)
        self.dem_reactive = (self.dem_reactive * c) / (self.power_base_mva * 10 ** 6)
        self.dem_active_std = (self.dem_active_std * c) / (self.power_base_mva * 10 ** 6)

        self.bus['max_gen_p_pu'] = (self.bus['max_gen_p'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['min_gen_p_pu'] = (self.bus['min_gen_p'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['max_gen_q_pu'] = (self.bus['max_gen_q'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['min_gen_q_pu'] = (self.bus['min_gen_q'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['max_pv_pu'] = (self.bus['max_pv'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['min_pv_pu'] = (self.bus['min_pv'] * c) / (self.power_base_mva * 10 ** 6)

        self.pv_std = (self.pv_std.values.T * self.bus['max_pv_pu'].values).T
        self.bus['ramping'] = (self.bus['ramping'] * c) / (self.power_base_mva * 10 ** 6)

        self.bus['min_storage'] = (self.bus['min_storage'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['max_storage'] = (self.bus['max_storage'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['max_charge'] = (self.bus['max_charge'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['max_discharge'] = (self.bus['max_discharge'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['init_storage'] = (self.bus['init_storage'] * c) / (self.power_base_mva * 10 ** 6)
        self.bus['final_storage'] = (self.bus['final_storage'] * c) / (self.power_base_mva * 10 ** 6)

        return c / (self.power_base_mva * 10 ** 6), self.power_base_mva * 10 ** 6 / c, z ** (-1)

    def get_bus_lines(self, bus_id):
        return self.lines.loc[(self.lines['from_bus'] == bus_id) | (self.lines['to_bus'] == bus_id)]

    def get_connectivity_mat(self, param=''):
        mat = np.zeros((self.n_bus, self.n_bus))

        start_indices = np.searchsorted(self.bus.index, self.lines.loc[:, 'from_bus'])
        end_indices = np.searchsorted(self.bus.index, self.lines.loc[:, 'to_bus'])
        if param:
            mat_values = self.lines[param]
        else:
            mat_values = 1

        mat[start_indices, end_indices] = mat_values
        return mat

    def construct_bus_pumps_mat(self):
        mat = np.zeros((len(self.bus), int(self.bus['pump_id'].max() + 1)))

        # Get row and column indices where the bus are connected to pumps (pump_id col is not Nan)
        row_indices = self.bus.index[~self.bus['pump_id'].isna()].to_numpy()
        col_indices = self.bus['pump_id'].dropna().astype(int).to_numpy()

        # Use row and column indices to set corresponding elements in the matrix to 1
        mat[row_indices, col_indices] = 1
        return mat

    def construct_generators_params(self):
        self.bus = pd.merge(self.bus, self.generators, left_index=True, right_index=True, how='outer')
        self.bus[['gen_a', 'gen_b', 'gen_c']] = self.bus[['gen_a', 'gen_b', 'gen_c']].fillna(0)
        self.bus['min_gen_p'] = self.bus[['min_gen_p']].fillna(0)
        self.bus['max_gen_p'] = self.bus[['max_gen_p']].fillna(0)
        self.bus['min_gen_q'] = self.bus[['min_gen_q']].fillna(0)
        self.bus['max_gen_q'] = self.bus[['max_gen_q']].fillna(0)
        self.bus['gen_type'] = self.bus[['gen_type']].fillna(0)
        self.bus['ramping'] = self.bus[['ramping']].fillna(10 ** 6)

    def construct_pv_params(self):
        if not self.pv.empty:
            self.bus = pd.merge(self.bus, self.pv, left_index=True, right_index=True, how='outer',
                                suffixes=('_gen', '_pv'))
            self.bus['ramping'] = self.bus['ramping_pv'].combine_first(self.bus['ramping_gen'])
            self.bus = self.bus.drop(columns=['ramping_gen', 'ramping_pv'])
            self.bus['min_pv'] = self.bus['min_pv'].fillna(0)
            self.bus['max_pv'] = self.bus['max_pv'].fillna(0)
        else:
            self.bus['min_pv'] = 0
            self.bus['max_pv'] = 0

    def construct_batteries_params(self):
        self.bus = pd.merge(self.bus, self.batteries, left_index=True, right_index=True, how='outer')
        self.bus[self.batteries.columns] = self.bus.infer_objects(copy=False)[self.batteries.columns].fillna(0)

    def get_pv_availability(self):
        max_power = np.multiply(self.bus['max_gen_p_pu'].values, self.max_gen_profile.T).T.values
        for i in range(self.n_bus):
            if self.bus.loc[i, 'gen_type'] != 'pv':
                max_power[i, :] = 0

        return max_power

    def get_admittance_mat(self):
        y = np.zeros((self.n_bus, self.n_bus))

        from_indices = self.lines['from_bus'].values
        to_indices = self.lines['to_bus'].values

        y[from_indices, to_indices] = -(1 / self.lines['x_pu'].values)
        y[to_indices, from_indices] = -(1 / self.lines['x_pu'].values)

        # Create the diagonal elements
        for bus in range(self.n_bus):
            y[bus, bus] = -y[bus].sum()

        return y

    def make_battery_columns_for_dcpf(self, bat_param):
        bat_columns = utils.get_mat_for_type(self.bus, self.batteries)
        bat_columns = bat_columns[:, np.any(bat_columns, axis=0)]
        param_columns = self.bus[bat_param].values.reshape(-1, 1)
        columns = np.divide(bat_columns, param_columns, where=param_columns != 0)
        columns[np.abs(columns) < 10 ** -6] = 0
        return columns

    def restore_lines_flows(self, theta):
        from_buses = self.lines['from_bus'].values
        to_buses = self.lines['to_bus'].values

        # Calculate the number of time steps
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        n_time_steps = theta.shape[1]

        # Create index arrays for efficient slicing
        from_idx = from_buses[:, np.newaxis]
        to_idx = to_buses[:, np.newaxis]
        time_idx = np.arange(n_time_steps)

        # Extract the admittances for the lines
        line_admittances = self.y[from_buses, to_buses]

        # Calculate angle differences
        angle_diffs = theta[from_idx, time_idx] - theta[to_idx, time_idx]

        # Calculate power flows
        power_flows = line_admittances[:, np.newaxis] * angle_diffs

        return power_flows