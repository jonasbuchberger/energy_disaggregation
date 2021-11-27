import pandas as pd
import torch


def _normalized_disaggregation_error(y, y_hat):
    """
        Function that computes the normalized disaggregation error (NDE)

        Arguments:
            y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
            y (torch.Tensor) : Shape (B x T x M) ground truth targets
        Returns:
            NDE (float): normalized disaggregation error
    """
    return torch.sum((y_hat - y) ** 2) / torch.sum(y ** 2)


def _estimated_accuracy(y, y_hat):
    """
       Function that computes the estimated accuracy (EAC)

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
       Returns:
           EAC (float): estimated accuracy
    """
    return 1. - (y_hat - y).abs().sum(dim=1).mean() / (2. * y.abs().sum(dim=1)).mean()


def _compute_f1(s, s_hat):
    """
       Function that computes true positives (tp), false positives (fp) and
       false negatives (fn).

       Arguments:
           s_hat (torch.Tensor) : Shape (B x T x M) model state predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth states
       Returns:
           tp (int), fp (int), fn (int) : Tuple[int] containing tp, fp and fn
    """
    tp = torch.sum(s * s_hat).float()
    fp = torch.sum(torch.logical_not(s) * s_hat).float()
    fn = torch.sum(s * torch.logical_not(s_hat)).float()

    return tp, fp, fn


def example_f1_score(s, s_hat):
    """
       Function that computes the example-based F1-score (eb-F1)

       Arguments:
           s_hat (torch.Tensor) : Shape (B x T x M) model state predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth states
       Returns:
           eb-F1 (float): example-based F1-score
    """

    tp, fp, fn = _compute_f1(s, s_hat)
    numerator = 2 * tp
    denominator = torch.sum(s).float() + torch.sum(s_hat).float()
    return numerator / (denominator + 1e-12)


def _mae(y, y_hat):
    """
       Function that computes the mean absolute error (MAE)

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
       Returns:
           MAE (float): mean absolute error
    """
    return torch.mean(torch.abs(y - y_hat))


def per_appliance_metrics(y_hat, y, s_hat, s):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
       Returns:
           metrics (torch.Tensor): Shape (M x 4) Matrix containing each metric (F1, NDE, EAC, MAE) for each appliance
    """
    # assumes that y_hat is the median prediction and that y_hat.shape == y.shape
    assert y_hat.shape == y.shape
    assert s_hat.shape == s.shape
    tensors = []
    s_hat_clone = s_hat[:]
    s_hat_clone[s_hat < 0.5] = 0
    s_hat_clone[s_hat >= 0.5] = 1
    for appliance in range(y_hat.shape[-1]):
        # try:
        #     f1 = f1_score(s[..., appliance], s_hat_clone[..., appliance], average='samples', zero_division=0)
        # except ValueError:
        #     print(s_hat_clone[torch.logical_and(s_hat_clone > 0, s_hat_clone < 1)])
        f1 = example_f1_score(s[..., appliance], s_hat[..., appliance])
        nde = _normalized_disaggregation_error(y[..., appliance], y_hat[..., appliance])
        eac = _estimated_accuracy(y[..., appliance], y_hat[..., appliance])
        mae = _mae(y[..., appliance], y_hat[..., appliance])
        tensors.append(torch.tensor([f1, nde, eac, mae]))
    return torch.stack(tensors)


def mean_metrics(y_hat, y, s_hat, s):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
       Returns:
           metrics (torch.Tensor): Shape (4, ) Vector containing the mean of each metric (F1, NDE, EAC, MAE)
    """
    return torch.mean(per_appliance_metrics(y_hat, y, s_hat, s), dim=0)


def per_appliance_metrics_pandas(y_hat, y, s_hat, s, appliances=None, metrics=None):
    """
       Function that computes the F1-Score, NDE, EAC and MAE per appliance and returns
       the results as dataframe

       Arguments:
           y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
           y (torch.Tensor) : Shape (B x T x M) ground truth targets
           s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
           s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
           appliances (List[String]) : List of appliance names
           metrics (List[String]) : List of metrics
       Returns:
           frame (pandas.DataFrame): Shape (M x 4) Matrix containing each metric for each appliance
    """
    if metrics is None:
        metrics = ['F1', 'NDE', 'EAC', 'MAE']
    if appliances is None:
        appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']

    per_appl_tensor = per_appliance_metrics(y_hat, y, s_hat, s)
    frame = pd.DataFrame(columns=['Appliance'] + metrics)
    frame['Appliance'] = appliances
    frame = frame.set_index('Appliance')
    for i, app in enumerate(appliances):
        frame.loc[app] = per_appl_tensor[i]
    return frame


def mean_metrics_pandas(y_hat, y, s_hat, s, appliances=None, metrics=None):
    """
          Function that computes the F1-Score, NDE, EAC and MAE per appliance and returns
          the results as dataframe

          Arguments:
              y_hat (torch.Tensor) : Shape (B x T x M) model regression predictions
              y (torch.Tensor) : Shape (B x T x M) ground truth targets
              s_hat (torch.Tensor) : Shape (B x T x M) model on/off predictions
              s (torch.Tensor) : Shape (B x T x M) ground truth on/off targets
              appliances (List[String]) : List of appliance names
              metrics (List[String]) : List of metrics
          Returns:
              frame (pandas.DataFrame): Shape (M x 4) Matrix containing the mean of each metric for each appliance
   """
    if metrics is None:
        metrics = ['F1', 'NDE', 'EAC', 'MAE']
    if appliances is None:
        appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']

    frame = per_appliance_metrics_pandas(y_hat, y, s_hat, s, appliances, metrics)
    return frame.mean()
