from collections.abc import Iterable
from datetime import datetime
from itertools import product

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_blond import BLOND
from src.data.dataset_ukdale import UKDALE
from src.models.models import CNN1D, UNetNiLM
from src.models.test import test
from src.models.train import train
from src.visualization.plot_window import plot_window, plot_combined_window

"""
Reference implementation for the following power and state estimation algorithm:
UNet-NILM: A Deep Neural Network for Multi-tasks Appliances State Detection and Power Estimation in NILM
By author: Faustine et al.
Link to paper: https://dl.acm.org/doi/abs/10.1145/3427771.3427859
Link to GitHub: https://github.com/sambaiga/UNETNiLM
"""

def train_unet(dataset, num_layers=5, slide_size=50, window_size=100, batch_size=128, model_path=None,
               target_appliances=None, train_config=None):
    """
    Training function for UNetNILM

    Arguments:
        dataset (string) : Name of dataset to use
        num_layers (int) : Number of layers to use for the UNetBlock
        slide_size (int) : slide size to use for the dataset
        window_size (int) : Size of a window
        batch_size (int) : Batch size
        model_path (string) : If set start with pretrained model stored in model_path
        target_appliances (List[string]) : Names of appliances
        train_config (dict) : Arguments passed to src.models.train.train
        {
            epochs (int) : Number of epochs to train
            taus (torch.Tensor) : Taus to use
            optim (torch.optim.Optimizer) : Reference to optimizer
            optim_kwargs (dict) : Dictionary containing the kwargs for the optimizer
            scheduler (torch.scheduler.Scheduler) : Reference to scheduler
            scheduler_kwargs (dict) : Dictionary containing the kwargs for the scheduler
            device (string) : Which device to use
            logdir (string) : Directory where tensorboard will write
            early_stopping (int) : If set number of epochs when early stopping criterion will be checked
        }
    Returns:
        void
    """
    if train_config is None:
        train_config = {
            'epochs': 100,
            'taus': torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975]),
            'optim': torch.optim.Adam,
            'optim_kwargs': {'lr': 0.001, 'betas': (0.9, 0.98)},
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_kwargs': {'factor': 0.1, 'patience': 5, 'mode': 'max'},
            'device': 'cuda:0',
            'logdir': f'./logs/{dataset}_unet_{num_layers}lay_{slide_size}slide',
            'early_stopping': None
        }

    if dataset == 'UKDALE':
        target_appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge',
                             'microwave'] if target_appliances is None else target_appliances
        train_set = UKDALE(1, 'train', appliances=target_appliances, window_size=window_size, slide_size=slide_size)
        val_set = UKDALE(1, 'val', appliances=target_appliances, window_size=window_size, slide_size=50)
        test_set = UKDALE(1, 'test', appliances=target_appliances, window_size=window_size, slide_size=50)
    elif dataset == 'BLOND':
        target_appliances = ['Dell U2711', 'Epson EB-65950', 'Lenovo T420', 'Lenovo X230 i7',
                             'MacBook Pro 15 Mid-2014'] if target_appliances is None else target_appliances
        train_set = BLOND('train', appliances=target_appliances, window_size=window_size, slide_size=slide_size)
        val_set = BLOND('val', appliances=target_appliances, window_size=window_size, slide_size=50)
        test_set = BLOND('test', appliances=target_appliances, window_size=window_size, slide_size=50)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UNetNiLM(
        num_layers=num_layers,
        features_start=8,
        n_channels=1,
        num_classes=len(target_appliances),
        pooling_size=16,
        window_size=window_size,
        num_quantiles=len(train_config['taus']),
        dropout=0.1,
        d_model=128,
    )

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    train_config['model'] = model
    train_config['train_loader'] = train_loader
    train_config['val_loader'] = val_loader
    train_config['test_loader'] = test_loader

    train_config['train_means'] = torch.from_numpy(train_set.means)[:-1]
    train_config['train_stds'] = torch.from_numpy(train_set.stds)[:-1]

    train_config['val_means'] = torch.from_numpy(val_set.means)[:-1]
    train_config['val_stds'] = torch.from_numpy(val_set.stds)[:-1]

    model_path = train(**train_config)

    train_config['means'] = torch.from_numpy(test_set.means)[:-1]
    train_config['stds'] = torch.from_numpy(test_set.stds)[:-1]
    train_config['appliances'] = target_appliances
    model.load_state_dict(torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    test(**train_config)


def train_cnn(dataset, slide_size=50, window_size=100, batch_size=128, model_path=None):
    """
        Training function for CNN1D

        Arguments:
            dataset (string) : Name of dataset to use
            slide_size (int) : slide size to use for the dataset
            window_size (int) : Size of a window
            batch_size (int) : Batch size
            model_path (string) : If set start with pretrained model stored in model_path
        Returns:
            void
    """
    train_config = {
        'epochs': 100,
        'taus': torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975]),
        'optim': torch.optim.Adam,
        'optim_kwargs': {'lr': 0.001, 'betas': (0.9, 0.98)},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 5, 'mode': 'max'},
        'device': 'cuda:0',
        'logdir': f'./logs/{dataset}_cnn1d_{slide_size}slide/',
        'early_stopping': None
    }

    if dataset == 'UKDALE':
        target_appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']
        train_set = UKDALE(1, 'train', appliances=target_appliances, window_size=window_size, slide_size=slide_size)
        val_set = UKDALE(1, 'val', appliances=target_appliances, window_size=window_size, slide_size=50)
        test_set = UKDALE(1, 'test', appliances=target_appliances, window_size=window_size, slide_size=50)
    elif dataset == 'BLOND':
        target_appliances = ['Dell U2711', 'Epson EB-65950', 'Lenovo T420', 'Lenovo X230 i7', 'MacBook Pro 15 Mid-2014']
        train_set = BLOND('train', appliances=target_appliances, window_size=window_size, slide_size=slide_size)
        val_set = BLOND('val', appliances=target_appliances, window_size=window_size, slide_size=50)
        test_set = BLOND('test', appliances=target_appliances, window_size=window_size, slide_size=50)
    else:
        return

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = CNN1D(
        num_classes=len(target_appliances),
        pooling_size=16,
        window_size=window_size,
        n_quantiles=len(train_config['taus']),
        dropout=0.0
    )

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    train_config['model'] = model
    train_config['train_loader'] = train_loader
    train_config['val_loader'] = val_loader
    train_config['test_loader'] = test_loader

    train_config['train_means'] = torch.from_numpy(train_set.means)[:-1]
    train_config['train_stds'] = torch.from_numpy(train_set.stds)[:-1]

    train_config['val_means'] = torch.from_numpy(val_set.means)[:-1]
    train_config['val_stds'] = torch.from_numpy(val_set.stds)[:-1]

    model_path = train(**train_config)

    train_config['means'] = torch.from_numpy(test_set.means)[:-1]
    train_config['stds'] = torch.from_numpy(test_set.stds)[:-1]
    model.load_state_dict(torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    test(**train_config)


def vis_window(cnn1d_path=None, unet_path=None, dataset=None, target_appliances=None, window_threshold=200,
               additional_path=""):
    """
    Function that creates and saves disaggregation windows

    Arguments:
        cnn1d_path (string) : Path to CNN1D weights
        unet_path (string): Path to UNetNILM weights
        dataset (string) : Name of dataset to use
        target_appliances (List[string]) : Names of appliances
        window_threshold (int) : State threshold when a window will be used
        additional_path (string) : Addtional path information for a better ordering of windows
    Returns:
        void
    """
    train_config = {
        'taus': torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975]),
    }

    window_size = 100
    n_quantiles = 5
    test_set = None
    if dataset == 'UKDALE':
        target_appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge',
                             'microwave'] if target_appliances is None else target_appliances
        test_set = UKDALE(1, 'test', appliances=target_appliances, window_size=window_size, slide_size=50)
    elif dataset == 'BLOND':
        target_appliances = ['Dell U2711', 'Epson EB-65950', 'Lenovo T420', 'Lenovo X230 i7',
                             'MacBook Pro 15 Mid-2014'] if target_appliances is None else target_appliances
        test_set = BLOND('test', appliances=target_appliances, window_size=window_size, slide_size=50)
    train_config['means'] = torch.from_numpy(test_set.means)[:-1]
    train_config['stds'] = torch.from_numpy(test_set.stds)[:-1]

    windows_idx = []
    for i in range(0, len(test_set)):
        _, _, s = test_set[i]
        # if torch.sum(s) > window_threshold:
        #     windows_idx.append(i)
        if ((any(s[0] == 0) and any(s[0] == 1)) or ((any(s[1] == 0) and any(s[1] == 1)))) and torch.sum(
                s) > window_threshold:
            windows_idx.append(i)

    if cnn1d_path is not None:
        cnn1d = CNN1D(
            num_classes=len(target_appliances),
            pooling_size=16,
            window_size=window_size,
            n_quantiles=n_quantiles,
            dropout=0.0
        )
        cnn1d.load_state_dict(torch.load(cnn1d_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
        train_config['model'] = cnn1d

        for idx in tqdm(windows_idx):
            plot_window(test_set[idx], f'test_windows/{dataset}/{additional_path}/CNN1D_{idx}', target_appliances,
                        **train_config)

    if unet_path is not None:
        unet = UNetNiLM(
            num_layers=7,
            features_start=8,
            n_channels=1,
            num_classes=len(target_appliances),
            pooling_size=16,
            window_size=window_size,
            num_quantiles=n_quantiles,
            dropout=0.1,
            d_model=128,
        )
        unet.load_state_dict(torch.load(unet_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
        train_config['model'] = unet

        for idx in tqdm(windows_idx):
            plot_window(test_set[idx], f'test_windows/{dataset}/{additional_path}/UNET_{idx}', target_appliances,
                        **train_config)

    if unet_path is not None and cnn1d_path is not None:
        for idx in tqdm(windows_idx):
            plot_combined_window(test_set[idx], f'test_windows/{dataset}/{additional_path}/UNET_CNN1D_{idx}',
                                 target_appliances, unet,
                                 cnn1d,
                                 **train_config)


def grid_search_BLOND_unet(epochs=100, target_appliances=None, **hparams):
    """
    Function that performs grid search for UNetNILM on the BLOND dataset.
    The hyperparameters given by hyparams may be single value or an Iterable of
    values. If it is an Iterable, then each value will be tried with each other
    hyperparameter given as an Iterable.

    Arguments:
        epochs (int) : Number of epochs to train
        target_appliances (List[string]) : Name of appliances to use
        hparams (dict) : Dictionary containing the hyperparameters
        {
            num_layers (int or Iterable[int]) : Number of layers to use
            slide_size (int or Iterable[int]) : Slide sizes to use
            lr (float or Iterable[float]) : Learning rates to use
            window_size (int or Iterable[int]) : Window sizes to use
            batch_size (int or Iterable[int]) : Batch sizes to use
        }
    Returns:
        void
    """
    dataset = 'BLOND'
    num_layers = hparams['num_layers']
    slide_size = hparams['slide_size']
    lr = hparams['lr']
    window_size = hparams['window_size']
    batch_size = hparams['batch_size']

    key_order = [key for key, value in hparams.items() if isinstance(value, Iterable)]
    values = [value for key, value in hparams.items() if isinstance(value, Iterable)]
    configs = [dict(zip(key_order, value)) for value in product(*values)]
    tensorboard_hparams = hparams

    for config in configs:
        num_layers = config.get('num_layers', num_layers)
        slide_size = config.get('slide_size', slide_size)
        lr = config.get('lr', lr)
        window_size = config.get('window_size', window_size)
        batch_size = config.get('batch_size', batch_size)

        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tensorboard_hparams['num_layers'] = num_layers
        tensorboard_hparams['slide_size'] = slide_size
        tensorboard_hparams['lr'] = lr
        tensorboard_hparams['window_size'] = window_size
        tensorboard_hparams['batch_size'] = batch_size
        train_config = {
            'epochs': epochs,
            'taus': torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975]),
            'optim': torch.optim.Adam,
            'optim_kwargs': {'lr': lr, 'betas': (0.9, 0.98)},
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_kwargs': {'factor': 0.1, 'patience': 5, 'mode': 'max'},
            'device': 'cuda:0',
            'logdir': f'./logs/3apps_other/{dataset}_unet_{time}',
            'early_stopping': None,
            'hparams': tensorboard_hparams,
        }
        train_unet(dataset, num_layers, slide_size, window_size, batch_size, target_appliances=target_appliances,
                   train_config=train_config)


if __name__ == '__main__':
    grid_search_BLOND_unet(num_layers=7, slide_size=50, window_size=100, lr=[1e-3, 1e-5, 1e-7], epochs=20,
                           batch_size=int(2 ** 10))
