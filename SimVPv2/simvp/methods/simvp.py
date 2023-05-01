import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F
from simvp.models import SimVP_Model
from .base_method import Base_method
import wandb

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)

            loss = self.criterion(pred_y[batch_y != -777], batch_y[batch_y != -777])
            loss.backward()
            self.model_optim.step()
            if not self.by_epoch:
                self.scheduler.step()

            num_updates += 1
            loss_mean += loss.item()
            if wandb.run is not None:
                batch_y[batch_y != -777] = batch_y[batch_y != -777]*train_loader.dataset.std + train_loader.dataset.mean
                pred_y[batch_y != -777] = pred_y[batch_y != -777]*train_loader.dataset.std + train_loader.dataset.mean
                pred_y = pred_y[batch_y != -777]
                batch_y = batch_y[batch_y != -777]
                train_rmse = F.mse_loss(pred_y, batch_y).sqrt().item()

                wandb.log({"Train RMSE:": train_rmse})
            losses_m.update(loss.item(), batch_x.size(0))
            # if loss.item() > 1_000:
            #     print(batch_x.shape)

            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)
            batch_y[batch_y != -777] = batch_y[batch_y != -777]*vali_loader.dataset.std + vali_loader.dataset.mean
            pred_y[batch_y != -777] = pred_y[batch_y != -777]*vali_loader.dataset.std + vali_loader.dataset.mean
            pred_y = pred_y[batch_y != -777]
            batch_y = batch_y[batch_y != -777]
            loss = self.criterion(pred_y, batch_y)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()
                                                  ), [pred_y, batch_y], [preds_lst, trues_lst]))

            if i * batch_x.shape[0] > 1000:
                break
    
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().sqrt().item()))
            total_loss.append(loss.mean().sqrt().item())
            if wandb.run is not None:
                wandb.log({"Validation RMSE:": loss.mean().sqrt().item()})
        
        total_loss = np.average(total_loss)

        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        return preds, trues, total_loss

    def test_one_epoch(self, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)
        for batch_x, batch_y in test_pbar:
            pred_y = self._predict(batch_x.to(self.device))
            batch_y[batch_y != -777] = batch_y[batch_y != -777]*test_loader.dataset.std + test_loader.dataset.mean
            pred_y[batch_y != -777] = pred_y[batch_y != -777]*test_loader.dataset.std + test_loader.dataset.mean
            pred_y = pred_y[batch_y != -777]
            batch_y = batch_y[batch_y != -777]

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

            if wandb.run is not None:
                loss = self.criterion(pred_y, batch_y.to(self.device))
                wandb.log({"Test RMSE:": loss.mean().sqrt().item()})
        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
