import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import torch
import torch.nn.functional as F
import wandb

def eval_metrics(y_pred, y_gt, std, mean, denormalize=True, **kwargs):
    if denormalize:
        y_pred = y_pred[y_gt != -777]
        y_gt = y_gt[y_gt != -777]
        y_gt = y_gt*std + mean
        y_pred = y_pred*std + mean
    accs = calculate_accuracy(y_gt, y_pred)
    rmse = calculate_rmse(y_gt, y_pred)
    rrmse = relative_rmse(y_gt, y_pred)
    return accs, rmse

def calculate_accuracy(y_gt, y_pred, min_m=2, max_m=5):
    meters_eval = list(range(min_m, max_m+1))
    accs = []
    for meters in meters_eval:
        acc = _calculate_accuracy(y_gt, y_pred, meters=meters).item()
        accs.append(acc)
        print(f'Accuracy for {meters} m. range:{acc:.3f}')
        if wandb.run is not None:
            wandb.log({f"Accuracy {meters}m.":acc})
    return accs

def calculate_rmse(y_gt, y_pred):
    rmse = F.mse_loss(y_gt, y_pred).sqrt().item()
    print(f'RMSE for predicted frame: {rmse:.3f}')
    if wandb.run is not None:
            wandb.log({f"RMSE":rmse})
    return rmse

def relative_rmse(y_gt, y_pred):
    rmse = F.mse_loss(y_gt, y_pred).sqrt().item()
    mean_true = torch.mean(y_gt)

    rrmse = rmse / mean_true
    print(f'rRMSE for predicted frame: {rrmse*100.:.3f}%')
    if wandb.run is not None:
            wandb.log({f"rRMSE":rrmse})
    return rrmse

# Example usage
true_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
predicted_values = torch.tensor([1.1, 1.9, 3.2, 3.9, 5.1])

rrmse = relative_rmse(true_values, predicted_values)
print("Relative RMSE:", rrmse)



def _calculate_accuracy(y_gt, y_pred, meters=3, max_zsd=39):
    gt_labels = torch.zeros_like(y_gt)
    pred_labels = torch.zeros_like(y_pred)
    labels = (torch.arange(max_zsd//meters) * meters)
    
    # label data
    for i, label in enumerate(labels):
        if label == labels.max():
            gt_labels[(y_gt > label)] = i 
            pred_labels[(y_pred > label)] = i
        else:
            gt_labels[(y_gt > label) & (y_gt < label+meters)] = i
            pred_labels[(y_pred > label) & (y_pred < label+meters)] = i
    # Calculate metrics
    correct = torch.eq(pred_labels, gt_labels)
    accuracy = torch.mean(correct.float())
    
    return accuracy

def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)


def metric(pred, true, mean, std, metrics=['mae', 'mse'],
           clip_range=[0, 1], spatial_norm=False):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        spatial_norm (bool): Weather to normalize the metric by HxW.
    Returns:
        dict: evaluation results
    """
    pred = pred * std + mean
    true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr',]
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, true, spatial_norm)

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, true, spatial_norm)

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, true, spatial_norm)

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    for k, v in eval_res.items():
        eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
        eval_log += eval_str

    return eval_res, eval_log
