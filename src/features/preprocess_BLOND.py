import datetime
import json
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.__init__ import ROOT_DIR

START_TIME = '2016-10-01T00-00-00'
END_TIME = '2017-05-01T00-00-00'


def _read_real_power(medal_id, socket_id, days, dataset_dir):
    """ Opens a h5 container and returns the real power data as numpy array

    Args:
        medal_id (int): Medal identifier
        socket_id (int): Index of socket
        days (list): List of dates represented as strings
        dataset_dir (string): Path of the dataset location
    Returns:
        real_power (np.array): Array containing the real power measurements
    """
    real_power = np.empty(0)
    for day in days:
        medal_path = os.path.join(dataset_dir, day, f'medal-{medal_id}/')
        for file_name in os.listdir(medal_path):
            if 'summary' in file_name:
                filename = os.path.join(medal_path, file_name)
                data = h5py.File(filename, 'r')
                real_power = np.append(real_power, data[f'real_power{socket_id}'][:])
    return real_power


def _quantilize_real_power(real_power, window_size):
    """ Filters a real power array with a quantile filter

    Args:
        real_power (np.array): Array containing the real power measurements
        window_size (int): Window size for the quantile filter
    Returns:
        quantiled_real_power (np.array): Array containing the quantiled real power measurements
    """
    quantiled_real_power = np.empty_like(real_power)
    for i in range(0, len(real_power), window_size):
        window = real_power[i:i + window_size]
        q_50 = np.quantile(window, .5)
        quantiled_real_power[i:i + window_size] = q_50
    return quantiled_real_power


def _get_appliance_start_and_end(medal_id, socket_id):
    """ Returns the start and endtime of all appliances of a medal and socket combination

    Args:
        medal_id (int): Medal identifier
        socket_id (int): Index of socket
    Returns:
        df (pd.dataframe): Dataframe containing appliance name, starttime, endtime and both converted to an index
    """
    appliance_log_path = os.path.join(ROOT_DIR, 'data/BLOND/appliance_log.json')

    with open(appliance_log_path) as json_data:
        appliance_log = json.load(json_data)

    medal_log = appliance_log[f'MEDAL-{medal_id}']['entries']

    df = pd.DataFrame(columns=['name', 'start', 'end', 'start_idx', 'end_idx'])

    i = 0
    while i < len(medal_log):
        appliance = medal_log[i][f'socket_{socket_id}']['appliance_name']

        start_time = medal_log[i]['timestamp']
        end_time = medal_log[i + 1]['timestamp'] if i < len(medal_log) - 1 else END_TIME

        while i < len(medal_log) - 1 and appliance == medal_log[i + 1][f'socket_{socket_id}']['appliance_name']:
            end_time = medal_log[i + 2]['timestamp'] if i < len(medal_log) - 2 else END_TIME
            i += 1

        start_idx = time_to_index(start_time)
        end_idx = time_to_index(end_time)

        end_idx_data = time_to_index(END_TIME)
        if start_idx < end_idx_data:
            if end_idx > end_idx_data:
                end_time = END_TIME
                end_idx = end_idx_data

            df = df.append(
                pd.Series([appliance, start_time, end_time, start_idx, end_idx],
                          index=df.columns), ignore_index=True)

        i += 1

    return df


def _iter_through_appliances(socket_list):
    """ (Assumption 2): This function was used to test different thresholds and window sizes for the quantile filter
        Saves chosen thresholds and window sizes in a json file

    Args:
        socket_list (list): List of medal and socket combinations: [[2,3], [4,5]]
    Returns:
    """
    dataset_dir = os.path.join(ROOT_DIR, 'data/BLOND/BLOND-50')
    days = os.listdir(dataset_dir)
    days.remove('2016-09-30')
    if '.DS_Store' in days:
        days.remove('.DS_Store')

    days = sorted(days)

    appliance_json = {}
    file_path = os.path.join(ROOT_DIR, "data/BLOND/preprocessed/BLOND-50/appliance_meta.json")
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fd:
            if len(fd.readlines()) > 0:
                fd.seek(0)
                appliance_json = json.load(fd)
        shutil.move(file_path, file_path + ".tmp")

    window_size = None
    with open(file_path, 'w') as fd:
        for socket in socket_list:
            medal_id = socket[0]
            socket_id = socket[1]

            real_power = _read_real_power(medal_id, socket_id, days, dataset_dir)
            df = _get_appliance_start_and_end(medal_id, socket_id)

            for _, row in df.iterrows():

                appliance_name = row['name']
                if appliance_name is None:
                    continue
                if appliance_name in appliance_json.keys():
                    continue

                begin, end = row['start_idx'], row['end_idx']
                idx = int((end - begin) / 2)

                appliance_real_power = real_power[begin + idx:begin + idx + 7 * 60 * 60 * 24]
                plt.plot(appliance_real_power)
                plt.title(f"Real power of {appliance_name}")
                plt.show()

                appliance_threshold = None
                while appliance_threshold is None:
                    threshold = float(input(f"Power threshold for {appliance_name}: "))
                    window_size = int(input(f"Window size for {appliance_name}: "))

                    quantile_real_power = _quantilize_real_power(appliance_real_power, window_size)
                    appliance_on_off = np.zeros_like(quantile_real_power)
                    appliance_on_off[quantile_real_power > threshold] = 1

                    plt.figure(figsize=(6, 12))
                    plt.suptitle(f'{appliance_name} with p={threshold} w={window_size}')
                    plt.subplot(311)
                    plt.title('Real Power')
                    plt.plot(appliance_real_power)
                    plt.subplot(312)
                    plt.title('Quantile Real Power')
                    plt.plot(quantile_real_power)
                    plt.subplot(313)
                    plt.title('States')
                    plt.plot(appliance_on_off)
                    plt.show()

                    decision = None
                    while decision is None:
                        decision = input("Again (a)\tContinue (c)\tQuit (q) ")
                        if decision == "a":
                            continue
                        elif decision == "c":
                            appliance_threshold = threshold
                            window_size = window_size
                        elif decision == "q":
                            json.dump(appliance_json, fd, indent=4)
                            return
                        else:
                            decision = None

                appliance_json[appliance_name] = {"threshold": appliance_threshold,
                                                  "window_size": window_size}

        json.dump(appliance_json, fd, indent=4)


def preprocess_blond(socket_list):
    """ Function to preprocess medal and socket combinations with the previous chosen thresholds and window sizes

    Args:
        socket_list (list): List of medal and socket combinations: [[2,3], [4,5]]
    Returns:
    """
    dataset_dir = os.path.join(ROOT_DIR, 'data/BLOND/BLOND-50')
    processed_data_path = os.path.join(ROOT_DIR, 'data/BLOND/preprocessed/BLOND-50')
    os.makedirs(processed_data_path, exist_ok=True)
    days = os.listdir(dataset_dir)
    days.remove('2016-09-30')
    if '.DS_Store' in days:
        days.remove('.DS_Store')

    days = sorted(days)

    appliance_list = []
    for socket in tqdm(socket_list):
        medal_id = socket[0]
        socket_id = socket[1]

        df = get_appliance_meta(medal_id, socket_id)
        real_power = _read_real_power(medal_id, socket_id, days, dataset_dir)

        aggregate = np.array([])
        for index, row in df.iterrows():
            threshold = row['threshold']
            start_idx = row['start_idx']
            end_idx = row['end_idx']
            window_size = row['window_size']

            if threshold != -1 and (end_idx - start_idx) == (time_to_index(END_TIME) - time_to_index(START_TIME)):
                quantile_real_power = _quantilize_real_power(real_power, window_size)

                appliance_slice = quantile_real_power[start_idx:end_idx]
                on_off = np.empty_like(appliance_slice)
                on_off[appliance_slice > threshold] = 1
                on_off[appliance_slice < threshold] = 0

                appliance_list.append(str(row['name']))
                appliance_name = str(row['name']).replace(" ", "")
                np.savez(os.path.join(processed_data_path, f"quantile_{appliance_name}"),
                         power=quantile_real_power, mean=np.mean(quantile_real_power), std=np.std(quantile_real_power))

                # Add each channel to get the aggregated load
                aggregate = aggregate + quantile_real_power if aggregate.size else quantile_real_power

    # Calculate the quantiles of aggregate signal
    quantile_aggregate = _quantilize_real_power(aggregate, 10)

    np.savez(os.path.join(processed_data_path, "quantile_aggregated.npz"),
             power=quantile_aggregate,
             mean=np.mean(quantile_aggregate), std=np.std(quantile_aggregate))

    print(appliance_list)


def index_to_time(idx):
    """ Converts an index into a datetime representation

    Args:
        idx (int): Index of the timestamp
    Returns:
        time (datetime): Converted date
    """
    start_time = datetime.datetime.strptime(START_TIME, "%Y-%m-%dT%H-%M-%S")
    delta = datetime.timedelta(seconds=int(idx))
    time = start_time + delta

    return time


def time_to_index(time):
    """ Converts a datetime into the index representation

    Args:
        time (datetime): Date to convert
    Returns:
        idx (int): Index of the timestamp
    """
    start_time = datetime.datetime.strptime(START_TIME, "%Y-%m-%dT%H-%M-%S")
    if isinstance(time, str):
        time = datetime.datetime.strptime(time, "%Y-%m-%dT%H-%M-%S")

    delta = (time - start_time).total_seconds()
    idx = int(max(delta, 0))

    return idx


def get_appliance_meta(medal_id, socket_id):
    """ Returns the appliance meta information of a medal and socket combination needed for preprocessing

    Args:
        medal_id (int): Medal identifier
        socket_id (int): Index of socket
    Returns:
        df (pd.dataframe): Dataframe containing appliance name, starttime, endtime, both converted to an index,
                           threshold and window size
    """
    appliance_log_path = os.path.join(ROOT_DIR, 'data/BLOND/appliance_log.json')
    appliance_meta_path = os.path.join(ROOT_DIR, 'data/BLOND/preprocessed/BLOND-50/appliance_meta.json')

    with open(appliance_log_path) as json_data:
        appliance_log = json.load(json_data)

    with open(appliance_meta_path) as json_data:
        appliance_meta = json.load(json_data)

    medal_log = appliance_log[f'MEDAL-{medal_id}']['entries']

    df = pd.DataFrame(columns=['name', 'start', 'end', 'start_idx', 'end_idx', 'threshold', 'window_size'])

    i = 0
    while i < len(medal_log):
        appliance = medal_log[i][f'socket_{socket_id}']['appliance_name']

        start_time = medal_log[i]['timestamp']
        end_time = medal_log[i + 1]['timestamp'] if i < len(medal_log) - 1 else END_TIME

        while i < len(medal_log) - 1 and appliance == medal_log[i + 1][f'socket_{socket_id}']['appliance_name']:
            end_time = medal_log[i + 2]['timestamp'] if i < len(medal_log) - 2 else END_TIME
            i += 1

        start_idx = time_to_index(start_time)
        end_idx = time_to_index(end_time)

        end_idx_data = time_to_index(END_TIME)
        if start_idx < end_idx_data:
            if end_idx > end_idx_data:
                end_time = END_TIME
                end_idx = end_idx_data

            df = df.append(pd.Series([appliance, start_time, end_time, start_idx, end_idx,
                                      appliance_meta[str(appliance)]['threshold'],
                                      appliance_meta[str(appliance)]['window_size']],
                                     index=df.columns), ignore_index=True)

        i += 1

    return df


def get_appliance_durations(log):
    """ Builds two json files one with the measurement duration of each appliance with medal to socket information.
        One with the max duration of an individual appliance.

    Args:
        log (string): Path to the appliance_meta.json
    Returns:
    """
    with open(log, 'r') as fd:
        appliance_metas = json.load(fd)

    res = {}
    max_durations = {}
    for medal_id in range(1, 16):
        for socket_id in range(1, 7):
            df = _get_appliance_start_and_end(medal_id, socket_id)
            for _, row in df.iterrows():
                appliance_name = row['name']
                if appliance_name is None:
                    continue
                if appliance_name not in appliance_metas.keys():
                    continue
                if appliance_metas[appliance_name]['threshold'] < 0:
                    continue
                begin, end = row['start_idx'], row['end_idx']
                diff = end - begin
                if appliance_name in res.keys():
                    res[appliance_name].update({medal_id: {socket_id: diff}})
                else:
                    res[appliance_name] = {medal_id: {socket_id: diff}}

                if appliance_name in max_durations.keys():
                    max_durations[appliance_name] = max(max_durations[appliance_name], diff)
                else:
                    max_durations[appliance_name] = diff

    with open(log + ".durations", 'w') as fd:
        json.dump(res, fd, indent=4)
    with open(log + ".max_durations", 'w') as fd:
        json.dump(max_durations, fd, indent=4)


def filter_max_durations_and_plot_windows(log):
    """ Function used to determine if a appliance has a meaningful load profile.
        Stores a real power plot of each appliance.

    Args:
        log (string): Path to the appliance_meta.json.max_durations
    Returns:
    """
    dataset_dir = os.path.join(ROOT_DIR, 'data/BLOND/BLOND-50')
    days = os.listdir(dataset_dir)
    days.remove('2016-09-30')
    if '.DS_Store' in days:
        days.remove('.DS_Store')

    days = sorted(days)

    with open(log, 'r') as fd:
        durations_log = json.load(fd)

    durations = list(durations_log.values())
    max_dur = max(durations)

    for medal_id in range(1, 16):
        for socket_id in range(1, 7):
            df = _get_appliance_start_and_end(medal_id, socket_id)
            for _, row in df.iterrows():
                appliance_name = row['name']
                if appliance_name is None:
                    continue
                if appliance_name not in durations_log.keys() or durations_log[appliance_name] < max_dur:
                    continue

                real_power = _read_real_power(medal_id, socket_id, days, dataset_dir)
                begin, end = row['start_idx'], row['end_idx']
                appliance_real_power = real_power[begin:end]
                plt.plot(appliance_real_power)
                plt.title(f"Real power of {appliance_name} for medal {medal_id} and socket {socket_id}")
                plt.savefig(os.path.join(ROOT_DIR, "figures", "BLOND_appliances_windows",
                                         f"{appliance_name}_{medal_id}_{socket_id}.png"))
                plt.clf()


if __name__ == '__main__':
    preprocess_blond([[8, 4], [12, 5], [4, 6], [6, 6], [2, 4]])
