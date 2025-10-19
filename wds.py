import os
import pandas as pd
import numpy as np
import yaml


class WDS:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.real_to_reactive = 0.75

        # parameters form params.yaml
        self.input_power_units = None
        self.desal_power = None

        # read other parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

        self.flows_factor = params["flows_factor"]

        self.combs = pd.read_csv(os.path.join(self.data_folder, 'combs.csv'), index_col=0)
        self.desal = pd.read_csv(os.path.join(self.data_folder, 'desalination.csv'), index_col=0)
        self.tanks = pd.read_csv(os.path.join(self.data_folder, 'tanks.csv'), index_col=0)
        self.demands = pd.read_csv(os.path.join(self.data_folder, 'demands.csv'), index_col=0)
        self.demands_std = pd.read_csv(os.path.join(self.data_folder, 'demands_std.csv'), index_col=0)
        self.tariffs = pd.read_csv(os.path.join(self.data_folder, 'tariffs.csv'), index_col=0)

        self.n_combs = len(self.combs)
        self.n_stations = self.combs["station"].nunique()
        self.n_desal = len(self.desal)
        self.n_tanks = len(self.tanks)

        self.scale()

        self.tanks['init_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['init_level'])
        self.tanks['min_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['min_level'])
        self.tanks['max_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['max_level'])

    def scale(self):
        self.combs['flow'] /= self.flows_factor
        self.desal['min_flow'] /= self.flows_factor
        self.desal['max_flow'] /= self.flows_factor
        self.demands /= self.flows_factor
        self.demands_std /= self.flows_factor
        self.desal_power *= self.flows_factor

    def get_tank_demand(self, tank_id):
        return self.demands.loc[:, tank_id]

    def level_to_vol(self, diameter, level):
        return level * np.pi * (diameter ** 2) / 4 / self.flows_factor
