import argparse


DATASET_CHOICES = [
    "semi-SUPPORT",
    "semi-METABRIC",
    "HFCR",
    "PBC",
    "GBM",
    "GBSG",
    "METABRIC",
    "NACD",
    "SUPPORT",
    "MIMIC-IV",
]

MODEL_CHOICES = [
    "DeepSurv",
    "N-MTLR",
    "AFTNN-Weibull",
    "AFTNN-LogLogistic",
    "DeepSurv-2B",
    "N-MTLR-2B",
    "AFTNN-Weibull-2B",
    "AFTNN-LogLogistic-2B",
    "DeepSurv-salad",
    "N-MTLR-salad",
    "AFTNN-Weibull-salad",
    "AFTNN-LogLogistic-salad",
    "Nnet-survival",
    "RSF",
    "GB",
    "DeepHit",
    "CoxTime",
    "IWSG",
    "SODEN",
    "CQRNN",
    "DCSurvival",
    "SurvivalBoost",
]


def str_to_bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str_to_list(arg):
    if arg is None or arg == "":
        return []
    return [int(x) for x in arg.split(",")]


def generate_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SALaD and paper benchmark survival models."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="SUPPORT",
        choices=DATASET_CHOICES,
        help="Dataset name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="N-MTLR-salad",
        choices=MODEL_CHOICES,
        help="Model name.",
    )
    parser.add_argument(
        "--n-exp",
        "--n_exp",
        dest="n_exp",
        type=int,
        default=10,
        help="Number of random train/validation/test splits.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")

    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Weight for SALaD orthogonality regularization.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Weight for SALaD IPM regularization.",
    )
    parser.add_argument(
        "--ipm",
        type=str,
        default="mmd-rbf",
        choices=["mmd2-lin", "mmd-lin", "mmd2-rbf", "mmd-rbf"],
        help="IPM discrepancy used by SALaD.",
    )
    parser.add_argument(
        "--e-dims",
        "--e_dims",
        dest="e_dims",
        type=str_to_list,
        default=[],
        help="Hidden dimensions for event distribution heads, comma-separated.",
    )
    parser.add_argument(
        "--c-dims",
        "--c_dims",
        dest="c_dims",
        type=str_to_list,
        default=[],
        help="Hidden dimensions for censoring distribution heads, comma-separated.",
    )

    parser.add_argument(
        "--neurons",
        type=str_to_list,
        default=[64, 64],
        help="Hidden dimensions for baseline networks, or SALaD representation networks.",
    )
    parser.add_argument(
        "--norm",
        type=str_to_bool,
        default=True,
        help="Whether to use batch normalization.",
    )
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout probability.")
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        help="Torch activation class name, e.g. ReLU, Tanh, ELU.",
    )
    parser.add_argument(
        "--n-quantiles",
        "--n_quantiles",
        dest="n_quantiles",
        type=int,
        default=9,
        choices=[4, 9, 19, 39, 49, 99],
        help="Number of quantiles for CQRNN.",
    )
    parser.add_argument(
        "--mono-method",
        "--mono_method",
        dest="mono_method",
        type=str,
        default="bootstrap",
        choices=["ceil", "floor", "bootstrap"],
        help="Method used to make CQRNN predictions monotonic.",
    )

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Torch optimizer name.")
    parser.add_argument(
        "--n-epochs",
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=10000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--early-stop",
        "--early_stop",
        dest="early_stop",
        type=str_to_bool,
        default=True,
        help="Whether to use validation early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        default=True,
        help="Whether to show training progress bars.",
    )

    return parser.parse_args()
