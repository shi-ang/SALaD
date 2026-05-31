import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from dcsurvival.nde import NDE
from utils.util_survival import extract_survival


class PhiInv(nn.Module):
    def __init__(self, phi):
        super(PhiInv, self).__init__()
        self.phi = phi

    def forward(self, y, max_iter=2000, tol=1e-6):
        with torch.no_grad():
            """
            # We will only run newton's method on entries which do not have
            # a manual inverse defined (via .inverse)
            inverse = self.phi.inverse(y)
            assert inverse.shape == y.shape
            no_inverse_indices = torch.isnan(inverse)
            # print(no_inverse_indices)
            # print(y[no_inverse_indices].shape)
            t_ = newton_root(
                self.phi, y[no_inverse_indices], max_iter=max_iter, tol=tol,
                t0=torch.ones_like(y[no_inverse_indices])*1e-10)

            inverse[no_inverse_indices] = t_
            t = inverse
            """
            t = newton_root(self.phi, y, max_iter=max_iter, tol=tol)

        topt = t.clone().detach().requires_grad_(True)
        f_topt = self.phi(topt)
        return self.FastInverse.apply(y, topt, f_topt, self.phi)

    class FastInverse(Function):
        '''
        Fast inverse function. To avoid running the optimization
        procedure (e.g., Newton's) repeatedly, we pass in the value
        of the inverse (obtained from the forward pass) manually.

        In the backward pass, we provide gradients w.r.t (i) `y`, and
        (ii) `w`, which are any parameters contained in PhiInv.phi. The
        latter is implicitly given by furnishing derivatives w.r.t. f_topt,
        i.e., the function evaluated (with the current `w`) on topt. Note
        that this should contain *values* approximately equal to y, but
        will have the necessary computational graph built up, but detached
        from y.
        '''

        @staticmethod
        def forward(ctx, y, topt, f_topt, phi):
            ctx.save_for_backward(y, topt, f_topt)
            ctx.phi = phi
            return topt

        @staticmethod
        def backward(ctx, grad):
            y, topt, f_topt = ctx.saved_tensors
            phi = ctx.phi

            with torch.enable_grad():
                # Call FakeInverse once again, in order to allow for higher
                # order derivatives to be taken.
                z = PhiInv.FastInverse.apply(y, topt, f_topt, phi)

                # Find phi'(z), i.e., take derivatives of phi(z) w.r.t z.
                f = phi(z)
                dev_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]

                # To understand why this works, refer to the derivations for
                # inverses. Note that when taking derivatives w.r.t. `w`, we
                # make use of autodiffs automatic application of the chain rule.
                # This automatically finds the derivative d/dw[phi(z)] which
                # when multiplied by the 3rd returned value gives the derivative
                # w.r.t. `w` contained by phi.
                return grad / dev_z, None, -grad / dev_z, None


def log_survival(t, shape, scale, risk):
    return -(torch.exp(
        risk + shape * torch.log(t) - shape * torch.log(scale)))  # used log transform to avoid numerical issue


def survival(t, shape, scale, risk):
    return torch.exp(log_survival(t, shape, scale, risk))


def log_density(t, shape, scale, risk):
    log_hazard = risk + shape * torch.log(t) - shape * torch.log(scale) \
                 + torch.log(1 / t) + torch.log(shape)
    return log_hazard + log_survival(t, shape, scale, risk)


# newtwon_root is used during phi_inverse
def newton_root(phi, y, t0=None, max_iter=2000, tol=1e-14, guarded=False):
    '''
    Solve
        f(t) = y
    using the Newton's root finding method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    t0: Tensor of shape `s` indicating the initial guess for the root.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the absolute difference |f(t) - y|.
        By default, this is set to 1e-14,
        beyond which instability could occur when using pytorch `DoubleTensor`.
    guarded: Whether we use guarded Newton's root finding method.
        By default False: too slow and is not necessary most of the time.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if t0 is None:
        t = torch.zeros_like(y)
    else:
        t = t0.clone().detach()

    s = y.size()
    for it in range(max_iter):

        with torch.enable_grad():
            f_t = phi(t.requires_grad_(True))
            fp_t = torch.autograd.grad(f_t.sum(), t)[0]
            assert not torch.any(torch.isnan(fp_t))

        assert f_t.size() == s
        assert fp_t.size() == s

        g_t = f_t - y

        # Terminate algorithm when all errors are sufficiently small.
        if (torch.abs(g_t) < tol).all():
            break

        if not guarded:
            t = t - g_t / fp_t
        else:
            step_size = torch.ones_like(t)
            for num_guarded_steps in range(2000):
                t_candidate = t - step_size * g_t / fp_t
                f_t_candidate = phi(t_candidate.requires_grad_(True))
                g_candidate = f_t_candidate - y
                overstepped_indices = torch.abs(g_candidate) > torch.abs(g_t)
                if not overstepped_indices.any():
                    t = t_candidate
                    print(num_guarded_steps)
                    break
                else:
                    step_size[overstepped_indices] /= 2.

    assert torch.abs(g_t).max() < tol, \
        "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
    assert t.size() == s
    return t


# Only sampling use bisection root
def bisection_root(phi, y, lb=None, ub=None, increasing=True, max_iter=100, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the bisection method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    lb, ub: lower and upper bounds for t.
    increasing: True if f is increasing, False if decreasing.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the difference in upper and lower bounds.
        By default, this is set to 1e-10,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if lb is None:
        lb = torch.zeros_like(y)
    if ub is None:
        ub = torch.ones_like(y)

    assert lb.size() == y.size()
    assert ub.size() == y.size()
    assert torch.all(lb < ub)

    f_ub = phi(ub)
    f_lb = phi(lb)
    assert torch.all(
        f_ub >= f_lb) or not increasing, 'Need f to be monotonically non-decreasing.'
    assert torch.all(
        f_lb >= f_ub) or increasing, 'Need f to be monotonically non-increasing.'

    assert (torch.all(
        f_ub >= y) and torch.all(
        f_lb <= y)) or not increasing, 'y must lie within lower and upper bound. max min y=%s, %s. ub, lb=%s %s' % (
    y.max(), y.min(), ub, lb)
    assert (torch.all(
        f_ub <= y) and torch.all(
        f_lb >= y)) or increasing, 'y must lie within lower and upper bound. y=%s, %s. ub, lb=%s %s' % (
    y.max(), y.min(), ub, lb)

    for it in range(max_iter):
        t = (lb + ub) / 2
        f_t = phi(t)

        if increasing:
            too_low, too_high = f_t < y, f_t >= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]
        else:
            too_low, too_high = f_t > y, f_t <= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]

        assert torch.all(ub - lb > 0. - tol), "lb: %s, ub: %s, tol: %s" % (lb, ub, tol)

    assert torch.all(ub - lb <= tol)
    return t


def bisection_default_increasing(phi, y, tol):
    '''
    Wrapper for performing bisection method when f is increasing.
    '''
    return bisection_root(phi, y, increasing=True, tol=tol)


def bisection_default_decreasing(phi, y):
    '''
    Wrapper for performing bisection method when f is decreasing.
    '''
    return bisection_root(phi, y, increasing=False)


class MixExpPhi(nn.Module):
    '''
    Sample net for phi involving the sum of 2 negative exponentials.
    phi(t) = m1 * exp(-w1 * t) + m2 * exp(-w2 * t)

    Network Parameters
    ==================
    mix: Tensor of size 2 such that such that (m1, m2) = softmax(mix)
    slope: Tensor of size 2 such that exp(m1) = w1, exp(m2) = w2

    Note that this implies
    i) m1, m2 > 0 and m1 + m2 = 1.0
    ii) w1, w2 > 0
    '''

    def __init__(self, init_w=None):
        import numpy as np
        super(MixExpPhi, self).__init__()

        if init_w is None:
            self.mix = nn.Parameter(torch.tensor(
                [np.log(0.2), np.log(0.8)], requires_grad=True))
            self.slope = nn.Parameter(
                torch.log(torch.tensor([1e1, 1e6], requires_grad=True)))
        else:
            assert len(init_w) == 2
            assert init_w[0].numel() == init_w[1].numel()
            self.mix = nn.Parameter(init_w[0])
            self.slope = nn.Parameter(init_w[1])

    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
        nquery, nmix = t.numel(), self.mix.numel()

        mix_ = torch.nn.functional.softmax(self.mix)
        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         torch.exp(self.slope)[None, :].expand(nquery, nmix))

        ret = torch.sum(mix_ * exps, dim=1)
        return ret.reshape(s)


class MixExpPhi2FixedSlope(nn.Module):
    # TODO: delete this class
    def __init__(self, init_w=None):
        super(MixExpPhi2FixedSlope, self).__init__()

        self.mix = nn.Parameter(torch.log(torch.tensor([0.25], requires_grad=True)))
        self.slope = torch.tensor([1e1, 1e6], requires_grad=True)

    def forward(self, t):
        z = 1. / (1 + torch.exp(-self.mix[0]))
        return z * torch.exp(-t * self.slope[0]) + (1 - z) * torch.exp(-t * self.slope[1])


class SurvivalCopula(nn.Module):
    # for known parametric survival marginals, e.g., Weibull distributions
    def __init__(self, phi, device, num_features, tol, hidden_size=32, max_iter=2000):
        super(SurvivalCopula, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        self.net_t = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.net_c = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.shape_t = nn.Parameter(torch.tensor(1.0))  # Event Weibull Shape
        self.scale_t = nn.Parameter(torch.tensor(1.0))  # Event Weibull Scale
        self.shape_c = nn.Parameter(torch.tensor(1.0))  # Censoring Weibull Shape
        self.scale_c = nn.Parameter(torch.tensor(1.0))  # Censoring Weibull Scale

    def forward(self, x, t, c, max_iter=2000):
        # the Covariates for Event and Censoring Model
        x_beta_t = self.net_t(x).squeeze()
        x_beta_c = self.net_c(x).squeeze()

        # In event density, censoring entries should be 0
        event_log_density = c * log_density(t, self.shape_t, self.scale_t, x_beta_t)
        censoring_log_density = (1 - c) * log_density(t, self.shape_c, self.scale_c, x_beta_c)

        S_E = survival(t, self.shape_t, self.scale_t, x_beta_t)
        S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
                S_E <= 1. + 1e-10).all(), "t %s, output %s" % (t, S_E,)
        assert (S_C >= 0.).all() and (
                S_C <= 1. + 1e-10).all(), "t %s, output %s" % (t, S_C,)

        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, max_iter=max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]

        logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1 - c) * torch.log(cur2)

        return torch.sum(logL)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))

        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur
        elif mode == 'cond_cdf':
            target_dims = others['cond_dims']

            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator / denominator


class DCSurvival(nn.Module):
    # with neural density estimators
    def __init__(self, phi, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        super(DCSurvival, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        self.sumo_e = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size],
                          layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)
        self.sumo_c = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size],
                          layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)
        self.time_bins = None
        self.to(device=device, dtype=torch.double)

    def forward(self, x, t, c, max_iter=2000):
        ref_param = next(self.parameters())
        x = x.to(device=ref_param.device, dtype=ref_param.dtype)
        t = t.to(device=ref_param.device, dtype=ref_param.dtype)
        c = c.to(device=ref_param.device, dtype=ref_param.dtype)
        S_E, density_E = self.sumo_e(x, t, gradient=True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x, t, gradient=True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
                S_E <= 1. + 1e-10).all(), "t %s, output %s" % (t, S_E,)
        assert (S_C >= 0.).all() and (
                S_C <= 1. + 1e-10).all(), "t %s, output %s" % (t, S_C,)

        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter=max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]

        logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1 - c) * torch.log(cur2)

        return torch.sum(logL)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))

        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur
        elif mode == 'cond_cdf':
            target_dims = others['cond_dims']

            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator / denominator

    def survival(self, X):
        with torch.no_grad():
            ref_param = next(self.parameters())
            X = X.to(device=ref_param.device, dtype=ref_param.dtype)
            time_bins = self.time_bins.to(device=ref_param.device, dtype=ref_param.dtype)
            survival_curves = X.new_empty((X.shape[0], len(time_bins)))
            for i in range(len(time_bins)):
                survival_curves[:, i] = self.sumo_e.survival(X, time_bins[i])
        return survival_curves

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            optimizer: str,
            batch_size: int,
            epochs: int,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        # self.reset_parameters()
        self.to(device=device, dtype=torch.double)

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer([{"params": self.sumo_e.parameters(), "lr": lr},
                           {"params": self.sumo_c.parameters(), "lr": lr},
                           {"params": self.phi.parameters(), "lr": lr}],
                          weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, y_train, _, _ = extract_survival(train_df, None)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            n_val = len(val_df)
            x_val, y_val, _, _ = extract_survival(val_df, None)
            x_val, y_val = x_val.to(device=device, dtype=torch.double), y_val.to(device=device, dtype=torch.double)
            t_val, e_val = y_val[:, 0], y_val[:, 1]

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device=device, dtype=torch.double), yi.to(device=device, dtype=torch.double)
                ti, ei = yi[:, 0], yi[:, 1]
                logloss = self(xi, ti, ei, max_iter=10000)
                loss = -logloss/len(xi)

                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation, do not use no_grad here becaue SuMo-net need to calculate the gradients
            self.eval()
            postfix = f"Train loss = {train_loss_ep:.4f};"

            if early_stop and not val_df.empty:
                logloss_val = self(x_val, t_val, e_val, max_iter=1000)
                eval_loss = -logloss_val / n_val
                # y_val_pred = self(x_val)
                # eval_loss = self.loss(y_val_pred, y_val)
                postfix += f" Val loss = {eval_loss:.4f};"

                if best_loss > eval_loss:
                    best_loss = eval_loss
                    best_ep = ep
                    torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                if (ep - best_ep) > patience:
                    postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                    pbar.set_postfix_str(postfix)
                    break
            pbar.set_postfix_str(postfix)
        # TODO: check all three parts of the model
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None


def sample(net, ndims, N, device, seed=142857):
    """
    Note: this does *not* use the efficient method described in the paper.
    Instead, we will use the naive method, i.e., conditioning on each
    variable in turn and then applying the inverse CDF method on the resultant conditional
    CDF.

    This method will work on all generators (even those defined by ACNet), and is
    the simplest method assuming no knowledge of the mixing variable M is known.
    """
    # Store old seed and set new seed
    old_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    # random variable generation
    U = torch.rand(N, ndims).to(device)

    for dim in range(1, ndims):
        print('Sampling from dim: %s' % dim)
        y = U[:, dim].detach().clone()

        def cond_cdf_func(u):
            U_ = U.clone().detach()
            U_[:, dim] = u
            cond_cdf = net.cond_cdf(U_[:, :(dim + 1)], "cond_cdf",
                                    others={'cond_dims': list(range(dim))})
            return cond_cdf

        # Call inverse using the conditional cdf `M` as the function.
        # Note that the weight parameter is set to None since `M` is not parameterized,
        # i.e., hardcoded as the conditional cdf itself.
        U[:, dim] = bisection_default_increasing(cond_cdf_func, y, tol=1e-8).detach()

    # Revert to old random state.
    torch.random.set_rng_state(old_rng_state)

    return U
