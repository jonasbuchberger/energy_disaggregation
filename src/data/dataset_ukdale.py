import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from src.__init__ import ROOT_DIR


class UKDALE(Dataset):

    def __init__(self, house_id, fold, appliances, window_size=100, slide_size=50):
        """ Dataset class for UKDALE

        Args:
            house_id (int): 1-5
            fold (string): train, val, test, all
            appliances (list): list containing appliance strings
            window_size (int): length of the returned windows
            slide_size (int): sliding size of the window
        """

        self.fold = fold
        self.appliances = appliances
        self.window_size = window_size
        self.slide_size = slide_size

        house_path = os.path.join(ROOT_DIR, f"data/UKDALE/preprocessed/house_{house_id}")
        labels_path = os.path.join(ROOT_DIR, f"data/UKDALE/house_{house_id}", 'labels.dat')
        meta_path = os.path.join(ROOT_DIR, f"data/UKDALE/metadata/building{house_id}.yaml")

        # Get channel_ids form given appliance strings
        labels = pd.read_csv(labels_path, delim_whitespace=True, header=None, index_col=[1],
                             names=['channel', 'appliance'])
        channels = labels.loc[self.appliances].values.flatten()

        # Load meta_data of given building
        with open(meta_path) as yaml_data:
            house_meta = yaml.full_load(yaml_data)

        means = np.zeros(len(channels) + 1)
        stds = np.zeros(len(channels) + 1)
        thresholds = np.zeros(len(channels))
        quantile_matrix = np.array([])
        for i, channel_id in enumerate(channels):

            # Finding appliance thresholds in metadata file
            # If appliances connected to a channel changes the user is notified
            for appliance in house_meta['appliances']:
                if channel_id in appliance['meters'] and 'on_power_threshold' in appliance.keys():
                    threshold = appliance['on_power_threshold']
                    if thresholds[i] == 0 or thresholds[i] == threshold:
                        thresholds[i] = threshold
                    else:
                        print(f'Multiple appliances thresholds in channel {channel_id}')

            # Appending quantiles of targets to matrix
            channel_path = os.path.join(house_path, f"quantile_resampled_channel_{channel_id}.npz")
            channel_data = np.load(channel_path)['power']
            quantile_matrix = np.vstack((quantile_matrix, channel_data)) if quantile_matrix.size else channel_data

            means[i] = np.load(channel_path)['mean']
            stds[i] = np.load(channel_path)['std']

        self.means = means
        self.stds = stds
        self.thresholds = thresholds
        self.quantile_matrix = quantile_matrix

        # Load the aggregated real_power signal with quantiles created during preprocessing
        aggregated_path = os.path.join(house_path, f"quantile_resampled_aggregated.npz")
        self.aggregated = np.load(aggregated_path)['power']
        self.means[-1] = np.load(aggregated_path)['mean']
        self.stds[-1] = np.load(aggregated_path)['std']

        l = len(self.aggregated)
        if self.fold == 'train':
            self.quantile_matrix = self.quantile_matrix[:, :int(l * 0.8)]
            self.aggregated = self.aggregated[:int(l * 0.8)]
        elif self.fold == 'val':
            self.quantile_matrix = self.quantile_matrix[:, int(l * 0.8):int(l * 0.9)]
            self.aggregated = self.aggregated[int(l * 0.8):int(l * 0.9)]
        elif self.fold == 'test':
            self.quantile_matrix = self.quantile_matrix[:, int(l * 0.9):]
            self.aggregated = self.aggregated[int(l * 0.9):]

    def __len__(self):
        """
        Returns:
            (int): number of windows
        """
        return int((self.quantile_matrix.shape[1] - self.window_size) / self.slide_size)

    def __getitem__(self, idx):
        """
        Arguments:
        - idx (int) : index of the image to return
        Returns:
        - aggregated_window (np.array): aggregated load signal with length window_size
        - target_window (np.array): matrix shape num_appliances X window_size
        - target_window (np.array): matrix shape num_appliances X window_size
        """

        idx = idx * self.slide_size

        # on_off_window: each row for one channel/appliance (0: appliance is off, 1: appliance is on)
        # appliance is on when its quantile real_power is higher than its threshold
        quantile_window = self.quantile_matrix[:, idx:idx + self.window_size]
        on_off_window = np.zeros_like(quantile_window)
        on_off_window[quantile_window > self.thresholds.reshape(-1, 1)] = 1

        aggregated_window = self.aggregated[idx:idx + self.window_size]

        # Normalize
        aggregated_window = (aggregated_window - self.means[-1]) / self.stds[-1]
        quantile_window = (quantile_window - self.means[:-1].reshape(-1, 1)) / self.stds[:-1].reshape(-1, 1)

        return torch.tensor(aggregated_window).float(), torch.tensor(quantile_window).float().T, torch.tensor(
            on_off_window).float().T

