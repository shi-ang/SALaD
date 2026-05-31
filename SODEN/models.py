from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict
from copy import deepcopy
from tqdm import trange
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint as odeint
from torch.optim.lr_scheduler import CosineAnnealingLR



class BaseSurvODEFunc(nn.Module):
    def __init__(self):
        super(BaseSurvODEFunc, self).__init__()
        self.nfe = 0
        self.batch_time_mode = False

    def set_batch_time_mode(self, mode=True):
        self.batch_time_mode = mode
        # `odeint` requires the output of `odefunc` to have the same size as
        # `init_cond` despite the how many steps we are going to evaluate. Set
        # `self.batch_time_mode` to `False` before calling `odeint`. However,
        # when we want to call the forward function of `odefunc` directly and
        # when we would like to evaluate multiple time steps at the same time,
        # set `self.batch_time_mode` to `True` and the output will have size
        # (len(t), size(y)).

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, y):
        raise NotImplementedError("Not implemented.")


class ODEFunc(BaseSurvODEFunc):
    def __init__(self, base_neural_net, num_features):
        super(ODEFunc, self).__init__()
        self.net = base_neural_net
        self.feature_size = num_features
        self.time_bins = None

    def forward(self, t, y):
        """
        Arguments:
          t: When self.batch_time_mode is False, t is a scalar indicating the
            time step to be evaluated. When self.batch_time_mode is True, t is
            a 1-D tensor with a single element [1.0].
          y: When self.batch_time_mode is False, y is a 1-D tensor with length
            2 + k, where the first dim indicates Lambda_t, the second dim
            indicates the final time step T to be evaluated, and the remaining
            k dims indicates the features. When self.batch_time_mode is True, y
            is a 2-D tensor with batch_size * (2 + k).
        """
        self.nfe += 1
        device = next(self.parameters()).device
        Lambda_t = y.index_select(-1, torch.tensor([0]).to(device)).view(-1, 1)
        T = y.index_select(-1, torch.tensor([1]).to(device)).view(-1, 1)
        x = y.index_select(-1, torch.tensor(range(2, y.size(-1))).to(device))
        # Rescaling trick
        # $\int_0^T f(s; x) ds = \int_0^1 T f(tT; x) dt$, where $t = s / T$
        inp = torch.cat(
            [Lambda_t,
             t.repeat(T.size()) * T,  # s = t * T
             x.view(-1, self.feature_size)], dim=1)
        output = self.net(inp) * T  # f(tT; x) * T
        zeros = torch.zeros_like(
            y.index_select(-1, torch.tensor(range(1, y.size(-1))).to(device))
        )
        output = torch.cat([output, zeros], dim=1)
        if self.batch_time_mode:
            return output
        else:
            return output.squeeze(0)

    def ode_loss(self, X_batch, Y_batch, D_batch):
        # for every data point, we integrate starting from time 0
        device = X_batch.device
        all_zeros = torch.zeros(X_batch.size(0), dtype=torch.float, device=device)

        init_cond = torch.cat([all_zeros.view(-1, 1),
                               Y_batch.view(-1, 1),
                               X_batch],
                              dim=1)

        # we integrate from 0 (time 0) to 1 (the observed time per data point)
        t = torch.tensor([0., 1.]).to(device)

        # here's code to call the ODE solver
        self.set_batch_time_mode(False)
        cumulative_hazards = odeint(self, init_cond, t, rtol=1e-4, atol=1e-8)[1:].squeeze()
        # note: the reason [1:].squeeze() shows up is as follows:
        # - the output of odeint is an iterable where the 0th element corresponds
        #   `t[0]` and the 1st element corresponds to `t[1]`
        # - by using the indexing `[1:]`, we are saying that we want the
        #   cumulative hazards at `t[1]`, which corresponds to the cumulative
        #   hazard evaluated at the observed time per data point
        # - after applying the indexing `[1;]`, the 0th dimension is no longer
        #   needed so we use `squeeze()` to get rid of it
        self.set_batch_time_mode(True)

        # now we can evaluate the ODE function to get the hazards
        hazards = self(t[1:], cumulative_hazards).squeeze()

        # some reformatting of the output
        cumulative_hazards = cumulative_hazards[:, 0]
        hazards = hazards[:, 0] / Y_batch.view(-1, 1)

        # use a clamp to avoid log 0
        log_hazards = torch.log(hazards.clamp(min=1e-8))

        # the loss is precisely L_{SODEN-NLL} in Section 5.2 of the monograph
        loss_batch = (-D_batch.view(-1, 1) * log_hazards
                      + cumulative_hazards).mean()
        return loss_batch

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            optimizer: str,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 10,
            verbose: bool = True
    ):
        self.to(device)

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)

        x_tr = torch.tensor(train_df.drop(columns=['time', 'event']).values, dtype=torch.float32)
        t_tr, e_tr = torch.tensor(train_df['time'].values, dtype=torch.float32), \
                        torch.tensor(train_df['event'].values, dtype=torch.float32)

        train_dataloader = DataLoader(TensorDataset(x_tr, t_tr, e_tr), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val = torch.tensor(val_df.drop(columns=['time', 'event']).values, dtype=torch.float32).to(device)
            t_val, e_val = torch.tensor(val_df['time'].values, dtype=torch.float32).to(device), \
                            torch.tensor(val_df['event'].values, dtype=torch.float32).to(device)
        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, ti, ei in train_dataloader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                loss = self.ode_loss(xi, ti, ei)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    eval_loss = self.ode_loss(x_val, t_val, e_val).item()
                    postfix += f" Val loss = {eval_loss:.4f};"

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        best_params = deepcopy(self.state_dict())
                        # torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(best_params) if early_stop and not val_df.empty else None
        # self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None

    def predict_cum_hazard(
            self,
            x: torch.Tensor,
    ):
        device = x.device
        with torch.no_grad():
            if self.time_bins[0] != 0:
                self.time_bins = torch.cat([torch.tensor([0.], dtype=torch.float32, device=device), self.time_bins])

            max_time = self.time_bins.max()
            time_bins_rescaled = self.time_bins / max_time

            # for every data point, we integrate starting from time 0 and we integrate
            # to the max time
            all_zeros = torch.zeros(x.size(0), dtype=torch.float, device=device)
            all_max = max_time * torch.ones(x.size(0), dtype=torch.float, device=device)

            init_cond = torch.cat([all_zeros.view(-1, 1),
                                   all_max.view(-1, 1),
                                   x],
                                  dim=1)

            # we integrate from 0 (time 0) to 1 (the observed time per data point)
            # t = torch.tensor(time_bins_rescaled, dtype=torch.float32).to(device)

            # here's code to call the ODE solver
            self.set_batch_time_mode(False)
            cumulative_hazards = odeint(self, init_cond, time_bins_rescaled, rtol=1e-4, atol=1e-8)
            self.set_batch_time_mode(True)

            # if inserted_0:
            #     # ignore the 0th time which is 0
            #     return cumulative_hazards[1:, :, 0]
            # else:
            return cumulative_hazards[:, :, 0]
            # output shape = (original time grid length, number of points)

    def predict_survival(
            self,
            x: torch.Tensor,
    ):
        cumulative_hazards = self.predict_cum_hazard(x)
        survival = torch.exp(-cumulative_hazards.T)
        return survival