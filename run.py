import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from args import generate_parser
from data import make_survival_data
from model.LatDecReps import CoxPH_LDR, LogLogisticAFT_LDR, MTLR_LDR, WeibullAFT_LDR
from model.Survival import CoxPH, CenQuanRegNN, LogLogisticAFT, MTLR, WeibullAFT
from model.TwoBranch import CoxPH2B, LogLogisticAFT2B, MTLR2B, WeibullAFT2B
from model.utils import build_sequential_nn
from SurvivalEVAL import QuantileRegEvaluator, SurvivalEvaluator
from utils import df2np, pad_tensor, print_performance, save_params, set_seed
from utils.util_survival import (
    format_pred_sksurv,
    make_mono_quantiles,
    make_time_bins,
    survival_data_split,
    xcal_from_hist,
)


CHECKPOINT_DIR = "logs"
DISCRETE_MODELS = {"N-MTLR", "N-MTLR-2B", "N-MTLR-salad", "DeepHit", "Nnet-survival", "IWSG"}
QUANTILE_MODELS = {"CQRNN"}


def _checkpoint_name(args, model):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cls_name = model.__class__.__name__
    safe_model_name = args.model.replace("/", "_")
    return os.path.join(CHECKPOINT_DIR, f"{safe_model_name}_{cls_name}_{args.timestamp}")


def _fit_torch_model(args, model, data_train, data_val, device):
    model.fit(
        data_train,
        data_val,
        device=device,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        lr=args.lr,
        lr_min=1e-3 * args.lr,
        weight_decay=args.weight_decay,
        early_stop=args.early_stop,
        fname=_checkpoint_name(args, model),
        verbose=args.verbose,
    )


def _predict_standard_survival(model, x_test, device, time_attr):
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    survival = model.predict_survival(x_test_tensor)
    return survival, getattr(model, time_attr)


def _run_deepsurv(args, data_train, data_val, x_test, device):
    if args.model == "DeepSurv":
        model = CoxPH(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    elif args.model == "DeepSurv-2B":
        model = CoxPH2B(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    else:
        model = CoxPH_LDR(
            n_features=args.n_features,
            rep_dims=args.neurons,
            event_dims=args.e_dims,
            censor_dims=args.c_dims,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
            ipm=args.ipm,
            alpha=args.alpha,
            beta=args.beta,
        )
    _fit_torch_model(args, model, data_train, data_val, device)
    return _predict_standard_survival(model, x_test, device, "time_bins")


def _run_n_mtlr(args, data_train, data_val, x_test, discrete_bins_e, discrete_bins_c, device):
    if args.model == "N-MTLR":
        model = MTLR(
            n_features=args.n_features,
            time_bins=discrete_bins_e,
            hidden_size=args.neurons,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
        )
    elif args.model == "N-MTLR-2B":
        model = MTLR2B(
            n_features=args.n_features,
            time_bins_event=discrete_bins_e,
            time_bins_censor=discrete_bins_c,
            hidden_size=args.neurons,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
        )
    else:
        model = MTLR_LDR(
            n_features=args.n_features,
            time_bins_event=discrete_bins_e,
            time_bins_censor=discrete_bins_c,
            rep_dims=args.neurons,
            event_dims=args.e_dims,
            censor_dims=args.c_dims,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
            ipm=args.ipm,
            alpha=args.alpha,
            beta=args.beta,
        )
    _fit_torch_model(args, model, data_train, data_val, device)
    survival, time_coordinates = _predict_standard_survival(model, x_test, device, "time_bins")
    return survival, pad_tensor(time_coordinates, 0, where="start")


def _run_weibull(args, data_train, data_val, x_test, device):
    if args.model == "AFTNN-Weibull":
        model = WeibullAFT(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    elif args.model == "AFTNN-Weibull-2B":
        model = WeibullAFT2B(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    else:
        model = WeibullAFT_LDR(
            n_features=args.n_features,
            rep_dims=args.neurons,
            event_dims=args.e_dims,
            censor_dims=args.c_dims,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
            ipm=args.ipm,
            alpha=args.alpha,
            beta=args.beta,
        )
    _fit_torch_model(args, model, data_train, data_val, device)
    return _predict_standard_survival(model, x_test, device, "t_grids")


def _run_loglogistic(args, data_train, data_val, x_test, device):
    if args.model == "AFTNN-LogLogistic":
        model = LogLogisticAFT(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    elif args.model == "AFTNN-LogLogistic-2B":
        model = LogLogisticAFT2B(args.n_features, args.neurons, args.norm, args.activation, args.dropout)
    else:
        model = LogLogisticAFT_LDR(
            n_features=args.n_features,
            rep_dims=args.neurons,
            event_dims=args.e_dims,
            censor_dims=args.c_dims,
            norm=args.norm,
            activation=args.activation,
            dropout=args.dropout,
            ipm=args.ipm,
            alpha=args.alpha,
            beta=args.beta,
        )
    _fit_torch_model(args, model, data_train, data_val, device)
    return _predict_standard_survival(model, x_test, device, "t_grids")


def _run_cqrnn(args, data_train, data_val, x_test, device, seed):
    model = CenQuanRegNN(
        n_features=args.n_features,
        hidden_size=args.neurons,
        n_quantiles=args.n_quantiles,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
        t_max=1.2 * data_train.time.max(),
    )
    _fit_torch_model(args, model, data_train, data_val, device)
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    quantiles = model.predict_quantiles(x_test_tensor)
    levels, quantiles = make_mono_quantiles(
        model.quan_levels.cpu().numpy(),
        quantiles.cpu().numpy(),
        method=args.mono_method,
        seed=seed,
    )
    return quantiles, levels


def _run_pycox_discrete(args, x_train, t_train, e_train, x_val, t_val, e_val, x_test, discrete_bins_e):
    import torchtuples as tt
    from pycox.models import DeepHitSingle, LogisticHazard

    labtrans = DeepHitSingle.label_transform(discrete_bins_e.numpy())
    net = tt.practical.MLPVanilla(
        in_features=args.n_features,
        num_nodes=args.neurons,
        out_features=labtrans.out_features,
        batch_norm=args.norm,
        dropout=args.dropout,
        activation=getattr(nn, args.activation),
    )
    optim = getattr(tt.optim, args.optimizer)
    if args.model == "DeepHit":
        model = DeepHitSingle(net, optim, device=args.device, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    else:
        model = LogisticHazard(net, optim, device=args.device, duration_index=labtrans.cuts)
    model.label_transform = labtrans

    y_train = model.label_transform.transform(t_train, e_train)
    fit_kwargs = {}
    callbacks = None
    if x_val is not None:
        y_val = model.label_transform.transform(t_val, e_val)
        fit_kwargs["val_data"] = (x_val, y_val)
        fit_kwargs["val_batch_size"] = x_val.shape[0]
        callbacks = [tt.callbacks.EarlyStopping()] if args.early_stop else None

    model.optimizer.set_lr(args.lr)
    model.optimizer.set("weight_decay", args.weight_decay)
    model.fit(
        input=x_train,
        target=y_train,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        callbacks=callbacks,
        verbose=args.verbose,
        **fit_kwargs,
    )
    survival_df = model.predict_surv_df(x_test)
    return survival_df.values.T, survival_df.index.values


def _run_coxtime(args, x_train, t_train, e_train, x_val, t_val, e_val, x_test):
    import torchtuples as tt
    from pycox.models import CoxTime
    from pycox.models.cox_time import MLPVanillaCoxTime

    labtrans = CoxTime.label_transform()
    labtrans.fit(t_train, e_train)
    net = MLPVanillaCoxTime(
        in_features=args.n_features,
        num_nodes=args.neurons,
        batch_norm=args.norm,
        dropout=args.dropout,
        activation=getattr(nn, args.activation),
    )
    model = CoxTime(net, getattr(tt.optim, args.optimizer), device=args.device, labtrans=labtrans)
    model.label_transform = labtrans
    y_train = model.label_transform.fit_transform(t_train, e_train)

    fit_kwargs = {}
    callbacks = None
    if x_val is not None:
        y_val = model.label_transform.transform(t_val, e_val)
        fit_kwargs["val_data"] = (x_val, y_val)
        fit_kwargs["val_batch_size"] = x_val.shape[0]
        callbacks = [tt.callbacks.EarlyStopping()] if args.early_stop else None

    model.optimizer.set_lr(args.lr)
    model.optimizer.set("weight_decay", args.weight_decay)
    model.fit(
        input=x_train,
        target=y_train,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        callbacks=callbacks,
        verbose=args.verbose,
        **fit_kwargs,
    )
    model.compute_baseline_hazards()
    survival_df = model.predict_surv_df(x_test)
    time_coordinates = np.concatenate([np.array([0]), survival_df.index.values], 0)
    survival = np.concatenate([np.ones([survival_df.values.T.shape[0], 1]), survival_df.values.T], 1)
    return survival, time_coordinates


def _run_soden(args, data_train, data_val, x_test, device):
    from SODEN.models import ODEFunc

    if args.neurons:
        layers = build_sequential_nn(args.n_features + 2, args.neurons, args.norm, args.activation, None)
        layers.append(nn.Linear(args.neurons[-1], 1))
        layers.append(nn.Softplus())
    else:
        layers = [nn.Linear(args.n_features + 2, 1), nn.Softplus()]
    model = ODEFunc(nn.Sequential(*layers).to(device), num_features=args.n_features)
    model.fit(
        data_train,
        data_val,
        device=device,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        lr=args.lr,
        lr_min=1e-3 * args.lr,
        weight_decay=args.weight_decay,
        early_stop=args.early_stop,
        verbose=args.verbose,
    )
    model.time_bins = torch.tensor(data_train["time"], dtype=torch.float).to(device).unique()
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    return model.predict_survival(x_test_tensor), model.time_bins


def _run_sksurv(args, x_train_val, t_train_val, e_train_val, x_test, seed):
    from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, RandomSurvivalForest

    y_train_val = np.empty(dtype=[("cens", bool), ("time", np.float64)], shape=t_train_val.shape[0])
    y_train_val["cens"] = e_train_val
    y_train_val["time"] = t_train_val
    if args.model == "RSF":
        model = RandomSurvivalForest(n_estimators=100, n_jobs=None, random_state=seed)
    else:
        model = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph", n_estimators=100, random_state=seed
        )
    model.fit(x_train_val, y_train_val)
    return format_pred_sksurv(model.predict_survival_function(x_test))


def _run_dcsurvival(args, data_train, data_val, x_test, device):
    from dcsurvival.dirac_phi import DiracPhi
    from dcsurvival.survival import DCSurvival

    phi = DiracPhi(depth=2, widths=[100, 100], lc_w_range=[0, 1.0], shift_w_range=[0.0, 2.0], device=device, tol=1e-14)
    model = DCSurvival(
        phi,
        device,
        num_features=args.n_features,
        tol=1e-14,
        hidden_size=args.neurons[0],
        hidden_surv=args.neurons[0],
    )
    _fit_torch_model(args, model, data_train, data_val, device)
    x_test_tensor = torch.from_numpy(x_test).double().to(device)
    time_coordinates = torch.tensor(data_train["time"], dtype=torch.double).to(device).unique()
    model.time_bins = time_coordinates
    return model.survival(x_test_tensor), time_coordinates


def _run_iwsg(args, data_train, data_val, x_test, discrete_bins_e, device):
    from iwsg.models import DiscNN
    from iwsg.wrapper import make_survival_prediction, train_iwsg

    n_bins = len(discrete_bins_e)
    f_model = DiscNN(args.n_features, n_bins, args.neurons, args.norm, args.activation, args.dropout)
    g_model = DiscNN(args.n_features, n_bins, args.neurons, args.norm, args.activation, args.dropout)
    f_model, _ = train_iwsg(
        data_train,
        data_val,
        f_model,
        g_model,
        bins=discrete_bins_e[1:],
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        lr=args.lr,
        lr_min=1e-3 * args.lr,
        weight_decay=args.weight_decay,
        device=device,
        early_stop=args.early_stop,
        fname=os.path.join(CHECKPOINT_DIR, f"IWSG_{args.timestamp}"),
        verbose=args.verbose,
    )
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    discrete_bins_e[0] = 0
    return make_survival_prediction(x_test_tensor, n_bins, f_model, device), discrete_bins_e


def _run_survival_boost(args, x_train_val, t_train_val, e_train_val, x_test, seed):
    from hazardous import SurvivalBoost

    y_train_val = np.empty(dtype=[("event", bool), ("duration", np.float64)], shape=t_train_val.shape[0])
    y_train_val["event"] = e_train_val
    y_train_val["duration"] = t_train_val
    model = SurvivalBoost(n_iter=200, show_progressbar=args.verbose, random_state=seed)
    model.fit(x_train_val, y_train_val)
    time_coordinates = np.unique(model.time_grid_)
    predicted_curves = model.predict_cumulative_incidence(x_test, times=time_coordinates)
    return predicted_curves[:, 0], time_coordinates


def _prepare_data(args):
    data, cols_stdz = make_survival_data(args.data, seed=args.seed)
    if "true_censor" in data.columns:
        data = data.drop(columns=["true_censor"])

    features = data.columns.to_list()
    if "true_time" in features:
        features.remove("true_time")
    if "time" not in data.columns or "event" not in data.columns:
        raise ValueError("Dataset must contain 'time' and 'event' columns.")

    cols_stdz = [col for col in cols_stdz if col in features]
    return data, features, cols_stdz


def _standardize_splits(data_train, data_val, data_test, features, cols_stdz):
    data_train = data_train.copy()
    data_val = data_val.copy()
    data_test = data_test.copy()

    if cols_stdz:
        scaler = StandardScaler()
        data_train.loc[:, cols_stdz] = scaler.fit_transform(data_train[cols_stdz])
        if not data_val.empty:
            data_val.loc[:, cols_stdz] = scaler.transform(data_val[cols_stdz])
        data_test.loc[:, cols_stdz] = scaler.transform(data_test[cols_stdz])

    data_train = data_train.astype("float32")[features]
    data_val = data_val.astype("float32")[features] if not data_val.empty else data_val
    data_test = data_test.astype("float32")[features]
    return data_train, data_val, data_test


def _split_and_transform(args, data, features, cols_stdz, seed):
    if args.early_stop:
        pct_train, pct_val, pct_test = 0.8, 0.1, 0.1
    else:
        pct_train, pct_val, pct_test = 0.9, 0.0, 0.1

    data_train, data_val, data_test = survival_data_split(
        data,
        stratify_colname="both",
        frac_train=pct_train,
        frac_val=pct_val,
        frac_test=pct_test,
        random_state=seed,
    )
    if args.data.startswith("semi-"):
        data_train = data_train.drop(columns=["true_time"])
        data_val = data_val.drop(columns=["true_time"])
        data_test = data_test.drop(columns=["time"]).rename(columns={"true_time": "time"})
        data_test.event = np.ones(data_test.shape[0])

    data_train, data_val, data_test = _standardize_splits(
        data_train, data_val, data_test, features, cols_stdz
    )
    data_train_val = pd.concat([data_train, data_val], ignore_index=True) if not data_val.empty else data_train
    return data_train, data_val, data_test, data_train_val


def _evaluate_prediction(model_name, prediction, t_test, e_test, t_train_val, e_train_val):
    if model_name in QUANTILE_MODELS:
        quantiles, levels = prediction
        evaluator = QuantileRegEvaluator(
            quantiles,
            levels,
            t_test,
            e_test,
            t_train_val,
            e_train_val,
            predict_time_method="Median",
            interpolation="Pchip",
        )
    else:
        survival, time_coordinates = prediction
        evaluator = SurvivalEvaluator(
            survival,
            time_coordinates,
            t_test,
            e_test,
            t_train_val,
            e_train_val,
            predict_time_method="Median",
            interpolation="Pchip",
        )

    c_index = evaluator.concordance()[0]
    ibs_score = evaluator.integrated_brier_score(num_points=10)
    hinge_abs = evaluator.mae(method="Hinge", verbose=False, weighted=False)
    po_abs = evaluator.mae(method="Pseudo_obs", verbose=False, weighted=True)
    km_cal_score = evaluator.km_calibration()
    dcal_pvalue, dcal_hist = evaluator.d_calibration()
    xcal_score = xcal_from_hist(dcal_hist)
    return c_index, ibs_score, hinge_abs, po_abs, km_cal_score, dcal_pvalue, xcal_score


def _run_model(
    args,
    data_train,
    data_val,
    x_train,
    t_train,
    e_train,
    x_val,
    t_val,
    e_val,
    x_test,
    x_train_val,
    t_train_val,
    e_train_val,
    seed,
    device,
):
    discrete_bins_e = discrete_bins_c = None
    if args.model in DISCRETE_MODELS:
        discrete_bins_e = make_time_bins(t_train, event=e_train)
        discrete_bins_c = make_time_bins(t_train, event=1 - e_train)
        if args.model in {"DeepHit", "Nnet-survival"}:
            discrete_bins_e[0] = float(max(t_train_val.min() - 1e-5, 0))
            discrete_bins_c[0] = float(max(t_train_val.min() - 1e-5, 0))

    if args.model in {"DeepSurv", "DeepSurv-2B", "DeepSurv-salad"}:
        return _run_deepsurv(args, data_train, data_val, x_test, device)
    if args.model in {"N-MTLR", "N-MTLR-2B", "N-MTLR-salad"}:
        return _run_n_mtlr(args, data_train, data_val, x_test, discrete_bins_e, discrete_bins_c, device)
    if args.model in {"AFTNN-Weibull", "AFTNN-Weibull-2B", "AFTNN-Weibull-salad"}:
        return _run_weibull(args, data_train, data_val, x_test, device)
    if args.model in {"AFTNN-LogLogistic", "AFTNN-LogLogistic-2B", "AFTNN-LogLogistic-salad"}:
        return _run_loglogistic(args, data_train, data_val, x_test, device)
    if args.model == "CQRNN":
        return _run_cqrnn(args, data_train, data_val, x_test, device, seed)
    if args.model in {"DeepHit", "Nnet-survival"}:
        return _run_pycox_discrete(args, x_train, t_train, e_train, x_val, t_val, e_val, x_test, discrete_bins_e)
    if args.model == "CoxTime":
        return _run_coxtime(args, x_train, t_train, e_train, x_val, t_val, e_val, x_test)
    if args.model == "SODEN":
        return _run_soden(args, data_train, data_val, x_test, device)
    if args.model in {"RSF", "GB"}:
        return _run_sksurv(args, x_train_val, t_train_val, e_train_val, x_test, seed)
    if args.model == "DCSurvival":
        return _run_dcsurvival(args, data_train, data_val, x_test, device)
    if args.model == "IWSG":
        return _run_iwsg(args, data_train, data_val, x_test, discrete_bins_e, device)
    if args.model == "SurvivalBoost":
        return _run_survival_boost(args, x_train_val, t_train_val, e_train_val, x_test, seed)
    raise ValueError(f"Unknown model name: {args.model}")


def main(args=None):
    if args is None:
        args = generate_parser()
    if not isinstance(args, argparse.Namespace):
        args = argparse.Namespace(**args)

    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.n_features = None
    device = torch.device(args.device)

    data, features, cols_stdz = _prepare_data(args)
    args.n_features = len(features) - 2
    path = save_params(args)

    ci = []
    mae_hinge = []
    mae_po = []
    ibs = []
    km_cal = []
    dcal_ps = []
    xcal_stats = []

    pbar_outer = trange(args.n_exp, disable=not args.verbose, desc="Experiment")
    for i in pbar_outer:
        seed = args.seed + i
        set_seed(seed, device)
        data_train, data_val, data_test, data_train_val = _split_and_transform(
            args, data, features, cols_stdz, seed
        )

        x_train, t_train, e_train = df2np(data_train)
        x_val, t_val, e_val = df2np(data_val) if not data_val.empty else (None, None, None)
        x_test, t_test, e_test = df2np(data_test)
        x_train_val, t_train_val, e_train_val = df2np(data_train_val)

        try:
            prediction = _run_model(
                args,
                data_train,
                data_val,
                x_train,
                t_train,
                e_train,
                x_val,
                t_val,
                e_val,
                x_test,
                x_train_val,
                t_train_val,
                e_train_val,
                seed,
                device,
            )
        except Exception as exc:
            if args.model == "SODEN":
                print(f"SODEN failed on split {i}: {exc}")
                continue
            raise

        scores = _evaluate_prediction(args.model, prediction, t_test, e_test, t_train_val, e_train_val)
        c_index, ibs_score, hinge_abs, po_abs, km_cal_score, dcal_pvalue, xcal_score = scores

        ci.append(c_index)
        ibs.append(ibs_score)
        mae_hinge.append(hinge_abs)
        mae_po.append(po_abs)
        km_cal.append(km_cal_score)
        dcal_ps.append(dcal_pvalue)
        xcal_stats.append(xcal_score)

    print_performance(
        path=path,
        Cindex=ci,
        IBS=ibs,
        MAE_Hinge=mae_hinge,
        MAE_PO=mae_po,
        KM_cal=km_cal,
        dcal_pvalues=dcal_ps,
        xCal_stats=xcal_stats,
    )


if __name__ == "__main__":
    main()
