import torch
import torch as t

Tensor = torch.FloatTensor


def MSE(y_hat, y, mask=None):
    if mask is None:
        mask = t.ones_like(y_hat)

    mse = (y - y_hat)**2
    mse = mask * mse
    mse = t.mean(mse)
    return mse


def MAELoss(y_hat, y, mask=None):
    if mask is None:
        mask = t.ones_like(y_hat)

    mae = t.abs(y - y_hat) * mask
    mae = t.mean(mae)
    return mae


def metrics3(arr_pred: Tensor, arr_true: Tensor):
    true_abs = arr_true.abs()
    wape = (arr_pred-arr_true).abs().sum() / true_abs.sum()
    nz = torch.where(true_abs > 0)
    Pnz = arr_pred[nz]
    Anz = arr_true[nz]
    mape = ((Pnz - Anz).abs() / (Anz.abs() + 1e-5)).mean()
    smape = (2 * (Pnz - Anz).abs() / (Pnz.abs() + Anz.abs() + 1e-5)).mean()
    return wape, mape, smape


def crps_ensemble(observations: Tensor, forecasts: Tensor) -> Tensor:
    '''pytorch version of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    
    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.
    '''
    assert observations.ndim == forecasts.ndim - 1
    assert observations.shape == forecasts.shape[1:]  # ensemble ~ first axis

    score = (forecasts - observations).abs().mean(axis=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    forecasts_diff = (torch.unsqueeze(forecasts, 0) -
                      torch.unsqueeze(forecasts, 1))
    score += -0.5 * forecasts_diff.abs().mean(axis=(0,1))
    return score