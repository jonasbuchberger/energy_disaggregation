import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.__init__ import ROOT_DIR

START_TIME = '2015-01-01T00-00-00'
END_TIME = '2015-04-01T00-00-00'
APPLIANCES = ['washing_machine', 'dishwasher', 'tv', 'kettle', 'fridge', 'microwave']
WINDOW_SIZES = {
    5: 50,
    6: 50,
    7: 50,
    10: 10,
    12: 50,
    13: 10,
}


def time_to_index(time):
    """
    Takes time string and converts it to an index from 1970/1/1

    Args:
        time (string): "%Y-%m-%dT%H-%M-%S"
    Returns:
        delta (int): time converted to index
    """

    time = datetime.strptime(time, "%Y-%m-%dT%H-%M-%S")
    delta = (time - datetime.utcfromtimestamp(0)).total_seconds()

    return int(delta)


def resample(start_idx, end_idx, data):
    """ (Assumption 1): Takes window of dataframe and resamples it with its mean.
        This way missing values are eliminated.

    Args:
        start_idx (int): start index of resample window
        end_idx (int): end index of resample window
        data (pd.Dataframe): dataframe containing real_power values and indices
    Returns:
        mean_value (float): mean of the window to be resampled
    """

    data_range = data.loc[start_idx:end_idx]

    # If data_range is empty increase end_index to resample missing values
    while data_range.empty:
        end_idx += 1
        data_range = data.loc[start_idx:end_idx]

    # Nanmean to eliminate Nan values
    mean_value = np.nanmean(data_range.values)

    return mean_value


def preprocessing(house_id=1):
    """
    Resamples each channel given by APPLIANCES

    Args:
        house_id (int): 1-5
    Returns:
        void
    """

    start_idx = time_to_index(START_TIME)
    end_idx = time_to_index(END_TIME)

    time_range_idx = np.arange(start_idx, end_idx, step=6)
    os.makedirs(os.path.join(ROOT_DIR, f"data/UKDALE/preprocessed/house_{house_id}/"), exist_ok=True)

    house_path = os.path.join(ROOT_DIR, f"data/UKDALE/house_{house_id}")
    house_meta_path = os.path.join(ROOT_DIR, f"data/UKDALE/metadata/building{house_id}.yaml")
    labels_path = os.path.join(house_path, 'labels.dat')

    labels = pd.read_csv(labels_path, delim_whitespace=True, header=None, index_col=[1],
                         names=['channel', 'appliance'])

    with open(house_meta_path) as json_data:
        house_meta = yaml.full_load(json_data)

    for channel_id in labels.loc[APPLIANCES].values.flatten():
        channel_path = os.path.join(house_path, f'channel_{channel_id}.dat')
        data = pd.read_csv(channel_path, delim_whitespace=True, header=None, names=['index', 'real_power'])
        data = data.set_index('index')
        resampled_appliance = np.zeros((len(time_range_idx),))
        window_size = 2
        latest = None
        for i, idx in tqdm(enumerate(time_range_idx)):
            if i + window_size >= len(time_range_idx):
                resampled_appliance[i] = latest
                continue
            new_value = resample(idx, time_range_idx[i + window_size], data)
            resampled_appliance[i] = new_value
            latest = new_value

        np.savez(os.path.join(ROOT_DIR, f"data/UKDALE/preprocessed/house_1/resampled_channel_{channel_id}.npz"),
                 resampled_appliance)


def preprocess_quantile(house_id=1):
    """
    Preprocesses the resampled data into quantiles

    Args:
        house_id (int): 1-5
    Returns:
        void
    """

    processed_data_path = os.path.join(ROOT_DIR, f"data/UKDALE/preprocessed/house_{house_id}/")
    os.makedirs(processed_data_path, exist_ok=True)

    aggregate = np.array([])
    for data in os.listdir(processed_data_path):
        if 'quantile' not in data:
            channel_path = os.path.join(processed_data_path, data)
            channel = np.load(channel_path)['arr_0']

            channel_id = int(channel_path.split('_')[3].split('.')[0])
            window_size = WINDOW_SIZES[channel_id]

            # Add each channel to get the aggregated load
            aggregate = aggregate + channel if aggregate.size else channel
            # Calculate quantiles of each resampled channel
            quantile_channel = _quantile_signal(channel, window_size)

            np.savez(os.path.join(processed_data_path, f"quantile_{data}"), power=quantile_channel,
                     mean=np.mean(quantile_channel), std=np.std(quantile_channel))

    # Calculate the quantiles of aggregate signal
    quantile_aggregate = _quantile_signal(aggregate, 10)

    np.savez(os.path.join(processed_data_path, "quantile_resampled_aggregated.npz"), power=quantile_aggregate,
             mean=np.mean(quantile_aggregate), std=np.std(quantile_aggregate))


def _quantile_signal(signal, window_size, quantile=.5):
    """
    Takes Signal and denoises it by using the quantile value of each window

    Args:
        signal (np.array): signal to create quantiles
        window_size (int): length of the quantile windows
        quantile (float): percentage of the calculate quantile
    Returns:
        quantile_signal (np.array): signal smoothed by quantiles
    """

    quantile_signal = np.empty_like(signal)

    for i in range(0, len(signal), window_size):
        window = signal[i:i + window_size]
        q = np.quantile(window, quantile)
        quantile_signal[i:i + window_size] = q

    return quantile_signal


def _normalize(signal):
    """
    Takes Signal and normalizes is by its mean and std

    Args:
        signal (np.array): signal to create quantiles
    Returns:
        normalized_signal (np.array): normalized signal
    """

    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std

    return normalized_signal


def undo_normalize(signal, mean, std):
    """
    Takes Signal and undoes the normalization with its mean and std

    Args:
        signal (np.array): signal to create quantiles
        mean (float): mean value used for normalizing
        std (float): standard deviation used for normalizing
    Returns:
        normalized_signal (np.array): normalized signal
    """
    return signal * std + mean


def index_to_time(index):
    import datetime
    delta = datetime.timedelta(seconds=index)
    from datetime import datetime
    return (datetime.utcfromtimestamp(0) + delta)


if __name__ == '__main__':
    preprocessing()
    preprocess_quantile()

