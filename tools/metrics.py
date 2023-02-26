import torch
import torch.nn.functional as F

def eval_metrics(y_gt, y_pred):
    calculate_accuracy(y_gt, y_pred)
    calculate_mse(y_gt, y_pred)

def calculate_accuracy(y_gt, y_pred, min_m=2, max_m=5):
    meters_eval = list(range(min_m, max_m+1))
    accs = []
    for meters in meters_eval:
        acc = _calculate_accuracy(y_gt, y_pred, meters=meters).item()
        accs.append(acc)
        print(f'Accuracy for {meters} m. range:{acc:.3f}')
    return accs

def calculate_mse(y_gt, y_pred):
    mse = F.mse_loss(y_gt, y_pred).sqrt().item()
    print(f'MSE for predicted frame: {mse:.3f}')
    return mse

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