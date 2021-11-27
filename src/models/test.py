import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import per_appliance_metrics_pandas
from ..features.preprocess_UKDALE import undo_normalize


def test(**train_config):
    """ Testing function

    Args:
        train_config (dict): Dictionary containing the parameters needed for training
         {
            model (nn.Module): Model to be trained
            test_loader (Dataloader): Dataloader containing test dataset
            taus (torch.Tensor): quantiles used for quantiled regression task
            logdir (str): Path to tensorboard logger
            device (str): Device used for training, either 'cuda:0' or 'cpu'
        }
    Returns:
        void
    """

    model = train_config['model']
    test_loader = train_config['test_loader']
    taus = train_config['taus']

    means = train_config['means'] if 'means' in train_config.keys() else 0
    stds = train_config['stds'] if 'stds' in train_config.keys() else 1

    num_test_its = len(test_loader)

    logger = SummaryWriter(train_config['logdir'])

    device = train_config['device'] if torch.cuda.is_available() else "cpu"
    appliances = train_config['appliances']

    model = model.to(device)
    taus = taus.to(device)
    quantile_idx = len(taus) // 2  # from model_pl, line 125

    val_running_metrics = 0
    model = model.eval()

    y_hats = []
    ys = []
    s_hats = []
    ss = []

    with torch.no_grad():
        for b in tqdm(test_loader):
            X, y, s = b
            X, y, s = X.to(device), y.to(device), s.to(device)
            y_hat, s_hat = model(X)

            y_hat = y_hat[:, :, quantile_idx, :].detach()
            s_hat = s_hat.detach()
            # val_running_metrics += (1. / num_test_its) * mean_metrics(y_hat, y, s_hat, s)

            y_hats.append(y_hat)
            ys.append(y)
            s_hats.append(s_hat)
            ss.append(s)

    y_hat, y, s_hat, s = torch.vstack(y_hats), torch.vstack(ys), torch.vstack(s_hats), torch.vstack(ss)
    y_hat, y = undo_normalize(y_hat, means.to(y_hat.device), stds.to(y_hat.device)), undo_normalize(y,
                                                                                                    means.to(y.device),
                                                                                                    stds.to(y.device))

    table = per_appliance_metrics_pandas(y_hat, y, s_hat, s, appliances=appliances)

    table.to_csv(os.path.join(train_config['logdir'], "metrics.csv"))
    mean_table = table.mean()

    logger.add_scalar('test/metrics/f1', mean_table['F1'])
    logger.add_scalar('test/metrics/nde', mean_table['NDE'])
    logger.add_scalar('test/metrics/eac', mean_table['EAC'])
    logger.add_scalar('test/metrics/mae', mean_table['MAE'])
    mean_table_dict = mean_table.to_dict()
    renamed_key_dict = {f'hparam/{k}': v for k, v in mean_table_dict.items()}
    logger.add_hparams(train_config['hparams'], renamed_key_dict)
    logger.close()
