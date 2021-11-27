import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import quantile_regression_loss
from .metrics import mean_metrics
from ..features.preprocess_UKDALE import undo_normalize


def train(**train_config):
    """ Training function

   Args:
        train_config (dict): Dictionary containing the parameters needed for training
        {
            epochs (int): Number of epochs of training
            model (nn.Module): Model to be trained
            train_loader (Dataloader): Dataloader containing training dataset
            val_loader (Dataloader): Dataloader containing validation dataset
            taus (torch.Tensor): quantiles used for quantiled regression task
            optim (torch.optim.Optimizer): Uninitialized optimizer, reference to optimizer (e.g. torch.optim.Adam, without brackets)
            optim_kwargs (dict): Dictionary containing the arguments for the optimizer
            logdir (str): Path to tensorboard logger
            device (str): Device used for training, either 'cuda:0' or 'cpu'
            scheduler (torch.optim.lr_scheduler):  Uninitialized scheduler, reference to scheduler
            scheduler_kwargs (dict): Dictionary containing the arguments for the scheduler

        }
    Returns:
        path (string) : Path to stored weights
    """

    epochs = train_config['epochs']
    model = train_config['model']
    train_loader = train_config['train_loader']
    val_loader = train_config['val_loader']
    taus = train_config['taus']
    optim = train_config['optim'](model.parameters(), **train_config['optim_kwargs'])
    early_stopping = train_config.get('early_stopping', None)
    scheduler = train_config['scheduler'](optim, **train_config['scheduler_kwargs'])

    train_means = train_config['train_means'] if 'train_means' in train_config.keys() else 0
    train_stds = train_config['train_stds'] if 'train_stds' in train_config.keys() else 1

    val_means = train_config['val_means'] if 'val_means' in train_config.keys() else 0
    val_stds = train_config['val_stds'] if 'val_stds' in train_config.keys() else 1

    early_stopping_val_loss = None
    best_loss = None

    num_train_its = len(train_loader)
    num_val_its = len(val_loader)

    logger = SummaryWriter(train_config['logdir'])

    device = train_config['device'] if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    taus = taus.to(device)
    quantile_idx = len(taus) // 2  # from model_pl, line 125

    for epoch in range(epochs):

        train_q_losses = []
        train_ce_losses = []
        train_losses = []
        train_running_metrics = 0

        model = model.train()

        y_hats = []
        ys = []
        s_hats = []
        ss = []

        for b in tqdm(train_loader):
            optim.zero_grad()
            X, y, s = b
            X, y, s = X.to(device), y.to(device), s.float().to(device)
            y_hat, s_hat = model(X)
            q_loss = quantile_regression_loss(y_hat, y, taus)
            ce_loss = F.binary_cross_entropy(torch.sigmoid(s_hat), s)
            loss = q_loss + ce_loss
            train_q_losses.append(q_loss)
            train_ce_losses.append(ce_loss)
            train_losses.append(loss)
            loss.backward()
            optim.step()

            y_hat = y_hat[:, :, quantile_idx, :].detach()
            s_hat = s_hat.detach()

            y_hats.append(y_hat.cpu())
            ys.append(y.cpu())
            s_hats.append(s_hat.cpu())
            ss.append(s.cpu())

            # train_running_metrics += (1. / num_train_its) * mean_metrics(y_hat, y, s_hat, s)

        y_hat, y, s_hat, s = torch.vstack(y_hats), torch.vstack(ys), torch.vstack(s_hats), torch.vstack(ss)
        y_hat, y = undo_normalize(y_hat, train_means.to(y_hat.device), train_stds.to(y_hat.device)) \
            , undo_normalize(y, train_means.to(y.device), train_stds.to(y.device))

        train_running_metrics = mean_metrics(y_hat, y, s_hat, s)

        logger.add_scalar('train/loss/quantile', torch.mean(torch.as_tensor(train_q_losses)), global_step=epoch)
        logger.add_scalar('train/loss/cross_entropy', torch.mean(torch.as_tensor(train_ce_losses)), global_step=epoch)
        logger.add_scalar('train/loss/loss', torch.mean(torch.as_tensor(train_losses)), global_step=epoch)

        logger.add_scalar('train/metrics/train_f1', train_running_metrics[0], global_step=epoch)
        logger.add_scalar('train/metrics/train_nde', train_running_metrics[1], global_step=epoch)
        logger.add_scalar('train/metrics/train_eac', train_running_metrics[2], global_step=epoch)
        logger.add_scalar('train/metrics/train_mae', train_running_metrics[3], global_step=epoch)

        val_q_losses = []
        val_ce_losses = []
        val_losses = []
        val_running_metrics = 0
        model = model.eval()

        y_hats = []
        ys = []
        s_hats = []
        ss = []

        with torch.no_grad():
            for b in val_loader:
                X, y, s = b
                X, y, s = X.to(device), y.to(device), s.to(device)
                y_hat, s_hat = model(X)
                q_loss = quantile_regression_loss(y_hat, y, taus)
                ce_loss = F.binary_cross_entropy(torch.sigmoid(s_hat), s)
                loss = q_loss + ce_loss
                val_q_losses.append(q_loss)
                val_ce_losses.append(ce_loss)
                val_losses.append(loss)
                y_hat = y_hat[:, :, quantile_idx, :].detach()
                s_hat = s_hat.detach()

                y_hats.append(y_hat.cpu())
                ys.append(y.cpu())
                s_hats.append(s_hat.cpu())
                ss.append(s.cpu())

                # val_running_metrics += (1. / num_val_its) * mean_metrics(y_hat, y, s_hat, s)

        val_loss = torch.mean(torch.as_tensor(val_losses))

        y_hat, y, s_hat, s = torch.vstack(y_hats), torch.vstack(ys), torch.vstack(s_hats), torch.vstack(ss)
        y_hat, y = undo_normalize(y_hat, val_means.to(y_hat.device), val_stds.to(y_hat.device)), undo_normalize(y,
                                                                                                                val_means.to(
                                                                                                                    y.device),
                                                                                                                val_stds.to(
                                                                                                                    y.device))

        val_running_metrics = mean_metrics(y_hat, y, s_hat, s)

        logger.add_scalar('val/loss/quantile', torch.mean(torch.as_tensor(val_q_losses)), global_step=epoch)
        logger.add_scalar('val/loss/cross_entropy', torch.mean(torch.as_tensor(val_ce_losses)), global_step=epoch)
        logger.add_scalar('val/loss/loss', val_loss, global_step=epoch)

        logger.add_scalar('val/metrics/f1', val_running_metrics[0], global_step=epoch)
        logger.add_scalar('val/metrics/nde', val_running_metrics[1], global_step=epoch)
        logger.add_scalar('val/metrics/eac', val_running_metrics[2], global_step=epoch)
        logger.add_scalar('val/metrics/mae', val_running_metrics[3], global_step=epoch)
        logger.add_scalar('hyperpara/learning_rate', optim.param_groups[0]['lr'], global_step=epoch)

        scheduler.step(val_running_metrics[0])

        if best_loss is None or best_loss < val_loss:
            torch.save(model.state_dict(), os.path.join(logger.log_dir, "model.pth"))
            best_loss = val_loss

        if early_stopping is not None and epoch % early_stopping == 0:
            if early_stopping_val_loss is not None and early_stopping_val_loss < val_loss:
                break
            else:
                early_stopping_val_loss = val_loss

    torch.save(model.state_dict(), os.path.join(logger.log_dir, "model.pth"))
    return os.path.join(logger.log_dir, "model.pth")
