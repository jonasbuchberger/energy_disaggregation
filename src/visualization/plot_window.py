import os

import matplotlib.pyplot as plt
import torch

from src.__init__ import ROOT_DIR
from ..features.preprocess_UKDALE import undo_normalize


def plot_window(sample, name, appliances, **train_config):
    """
    Function plotting and saving a disaggregation window where the first row shows ground truth
    and the second row the models disaggregation.

    Arguments:
        sample (Tuple[torch.Tensor]) : Sample from src.data.*.Dataset
        name (string) : Name of the outputfile
        appliances (List[string]) :  Appliance names for the legend of the plot
        train_config (dict) : Dictionary containing parameters from training
        {
            model (nn.Module) : Model to be used for disaggregation
            taus (torch.Tensor) : Taus used
            means (torch.Tensor) : Mean value of the dataset used
            stds (torch.Tensor) : Standard deviation of the dataset used
        }
    Returns:
        void
    """
    agg, y, s = sample
    agg = agg.unsqueeze(0)

    model = train_config['model']
    model.eval()
    taus = train_config['taus']
    quantile_idx = len(taus) // 2  # from model_pl, line 125

    y_pred, s_pred = model(agg)
    y_pred = y_pred.detach().squeeze()
    s_pred = s_pred.detach().squeeze()

    # undo normalize
    means, stds = train_config['means'].to(y_pred.device), train_config['stds'].to(y_pred.device)
    y_pred, y = undo_normalize(y_pred, means, stds), undo_normalize(y, means, stds)

    s_pred[s_pred >= 0] = 1
    s_pred[s_pred < 0] = 0

    y_pred = s_pred * y_pred[:, quantile_idx, :]

    y = y * s

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(y)
    axs[0, 0].set_title('Real Power')
    axs[0, 1].plot(s)
    # axs[0, 1].set_ylim((0, 1))
    axs[0, 1].set_title('States')
    axs[1, 0].plot(y_pred)
    axs[1, 0].set_title('Predicted Real Power')
    axs[1, 1].plot(s_pred)
    # axs[1, 1].set_ylim((0, 1))
    axs[1, 1].set_title('Predicted States')

    for ax in axs.flat:
        ax.label_outer()

    lgd = fig.legend(appliances, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3)

    path = os.path.join(ROOT_DIR, 'figures', name)
    os.makedirs(path[:path.rfind('/')], exist_ok=True)
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)


def plot_combined_window(sample, name, appliances, unet, cnn1d, **train_config):
    """
        Function plotting and saving a disaggregation window for UNetNILM and CNN1D where the first row shows ground truth
        and the second row CNN1d and the third row UNetNILM disaggregation.

        Arguments:
            sample (Tuple[torch.Tensor]) : Sample from src.data.*.Dataset
            name (string) : Name of the outputfile
            appliances (List[string]) :  Appliance names for the legend of the plot
            unet (nn.Module) : UNetNILM model
            cnn1d (nn.Module) : CNN1D model
            train_config (dict) :
            {
                taus (torch.Tensor) : Taus used
                means (torch.Tensor) : Mean value of the dataset used
                stds (torch.Tensor) : Standard deviation of the dataset used
            }
        Returns:
            void
    """
    agg, y, s = sample
    agg = agg.unsqueeze(0)

    unet = unet.eval()
    cnn1d = cnn1d.eval()

    quantile_idx = len(train_config['taus']) // 2
    with torch.no_grad():
        y_pred_unet, s_pred_unet = unet(agg)
        y_pred_cnn1d, s_pred_cnn1d = cnn1d(agg)

        y_pred_unet, s_pred_unet = y_pred_unet.squeeze().cpu(), s_pred_unet.squeeze().cpu()
        y_pred_cnn1d, s_pred_cnn1d = y_pred_cnn1d.squeeze().cpu(), s_pred_cnn1d.squeeze().cpu()

    means, stds = train_config['means'].to(y_pred_unet.device), train_config['stds'].to(y_pred_unet.device)
    y_pred_unet, y = undo_normalize(y_pred_unet, means, stds), undo_normalize(y, means, stds)
    y_pred_cnn1d = undo_normalize(y_pred_cnn1d, means, stds)

    s_pred_unet[s_pred_unet >= 0] = 1
    s_pred_unet[s_pred_unet < 0] = 0

    s_pred_cnn1d[s_pred_cnn1d >= 0] = 1
    s_pred_cnn1d[s_pred_cnn1d < 0] = 0

    y_pred_unet = s_pred_unet * y_pred_unet[:, quantile_idx, :]
    y_pred_cnn1d = s_pred_cnn1d * y_pred_cnn1d[:, quantile_idx, :]

    y *= s

    fig, axes = plt.subplots(3, 2, figsize=(7, 6))
    axes[0, 0].plot(y)
    axes[0, 0].set_title('Real power')
    axes[0, 0].set_ylabel('Ground truth')
    axes[0, 1].plot(s)
    axes[0, 1].set_title('States')

    axes[1, 0].plot(y_pred_cnn1d)
    axes[1, 0].set_ylabel('CNN1D')
    axes[1, 1].plot(s_pred_cnn1d)

    axes[2, 0].plot(y_pred_unet)
    axes[2, 0].set_ylabel('UNet')
    axes[2, 1].plot(s_pred_unet)

    for ax in axes.flat:
        ax.label_outer()

    lgd = fig.legend(appliances, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3)
    path = os.path.join(ROOT_DIR, 'figures', name)
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)
