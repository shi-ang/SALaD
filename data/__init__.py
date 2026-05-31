from __future__ import annotations

import pickle

import numpy as np
import pandas as pd


SUPPORTED_DATASETS = [
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


def make_survival_data(dataset: str, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    if dataset == "semi-SUPPORT":
        return make_semi_synth("SUPPORT", seed=seed, target_censor_frac=0.5)
    if dataset == "semi-METABRIC":
        return make_semi_synth("METABRIC", seed=seed, target_censor_frac=0.5)
    if dataset == "HFCR":
        return make_heart_failure()
    if dataset == "PBC":
        return make_pbc()
    if dataset == "GBM":
        return make_gbm()
    if dataset == "GBSG":
        return make_gbsg()
    if dataset == "METABRIC":
        return make_metabric()
    if dataset == "NACD":
        return make_nacd()
    if dataset == "SUPPORT":
        return make_support()
    if dataset == "MIMIC-IV":
        return make_mimic_iv()
    raise ValueError(
        f"Unknown dataset '{dataset}'. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}."
    )


def make_support() -> tuple[pd.DataFrame, list[str]]:
    cols_to_drop = [
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
        "sfdm2",
        "adlp",
        "adls",
        "dzgroup",
        "sps",
        "aps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "hday",
    ]
    data = (
        pd.read_csv("data/support2.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"d.time": "time", "death": "event"})
    )
    data["event"] = data["event"].astype(int)
    data["ca"] = (data["ca"] == "metastatic").astype(int)

    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        "edu": data["edu"].mean(),
        "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(),
        "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(),
        "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(),
        "temp": data["temp"].mean(),
        "sod": data["sod"].mean(),
        "income": data["income"].mode()[0],
        "race": data["race"].mode()[0],
    }
    data = data.fillna(fill_vals)

    with pd.option_context("future.no_silent_downcasting", True):
        data.sex = data.sex.replace({"male": 1, "female": 0}).infer_objects(copy=False)
        data.income = data.income.replace(
            {"under $11k": 0, "$11-$25k": 1, "$25-$50k": 2, ">$50k": 3}
        ).infer_objects(copy=False)

    skip_cols = ["event", "sex", "time", "dzclass", "race", "diabetes", "dementia", "ca"]
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    data = pd.get_dummies(data, columns=["dzclass", "race"], drop_first=True)
    data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})
    data.reset_index(drop=True, inplace=True)
    return data, cols_standardize


def make_nacd() -> tuple[pd.DataFrame, list[str]]:
    cols_to_drop = ["PERFORMANCE_STATUS", "STAGE_NUMERICAL", "AGE65"]
    data = (
        pd.read_csv("data/NACD_full.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"delta": "event"})
    )
    data = data.drop(data[data["time"] <= 0].index).reset_index(drop=True)
    cols_standardize = [
        "BOX1_SCORE",
        "BOX2_SCORE",
        "BOX3_SCORE",
        "BMI",
        "WEIGHT_CHANGEPOINT",
        "AGE",
        "GRANULOCYTES",
        "LDH_SERUM",
        "LYMPHOCYTES",
        "PLATELET",
        "WBC_COUNT",
        "CALCIUM_SERUM",
        "HGB",
        "CREATININE_SERUM",
        "ALBUMIN",
    ]
    return data, cols_standardize


def make_metabric() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/Metabric.csv").rename(columns={"delta": "event"})
    cols_standardize = [
        "age_at_diagnosis",
        "size",
        "lymph_nodes_positive",
        "stage",
        "lymph_nodes_removed",
        "NPI",
    ]
    return data, cols_standardize


def make_gbsg() -> tuple[pd.DataFrame, list[str]]:
    data = (
        pd.read_csv("data/GBSG.csv")
        .drop(["pid"], axis=1)
        .rename(columns={"status": "event", "rfstime": "time"})
    )
    cols_standardize = ["age", "size", "grade", "nodes", "pgr", "er"]
    return data, cols_standardize


def make_gbm() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    data.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)
    data = data[data.time.notna()]
    data = data.drop(data[data["time"] <= 0].index).reset_index(drop=True)

    with pd.option_context("future.no_silent_downcasting", True):
        data.gender = data.gender.replace({"male": 1, "female": 0}).infer_objects(copy=False)
        data.radiation_therapy = data.radiation_therapy.replace(
            {"yes": 1, "no": 0}
        ).infer_objects(copy=False)
        data.ethnicity = data.ethnicity.replace(
            {"not hispanic or latino": 0, "hispanic or latino": 1}
        ).infer_objects(copy=False)

    data = pd.get_dummies(data, columns=["histological_type", "race"], drop_first=True)
    data = data.fillna(
        {
            "radiation_therapy": data["radiation_therapy"].median(),
            "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
            "ethnicity": data["ethnicity"].median(),
        }
    )
    data.columns = data.columns.str.replace(" ", "_")
    cols_standardize = [
        "years_to_birth",
        "date_of_initial_pathologic_diagnosis",
        "karnofsky_performance_score",
    ]
    return data, cols_standardize


def make_mimic_iv() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/MIMIC_IV_all_cause_failure.csv")
    skip_cols = [
        "event",
        "is_male",
        "time",
        "is_white",
        "renal",
        "cns",
        "coagulation",
        "cardiovascular",
    ]
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_pbc() -> tuple[pd.DataFrame, list[str]]:
    with open("data/cirrhosis.pkl", "rb") as f:
        cirrhosis = pickle.load(f)

    data = cirrhosis.data.original.drop(["ID"], axis=1).rename(
        columns={"Status": "event", "N_Days": "time"}
    )
    with pd.option_context("future.no_silent_downcasting", True):
        data = data.replace({"NaNN": np.nan}).infer_objects(copy=False)
        data.event = data.event.replace({"C": 0, "CL": 0, "D": 1}).infer_objects(copy=False)
        data.Drug = data.Drug.replace(
            {"D-penicillamine": 0, "Placebo": 1}
        ).infer_objects(copy=False)
        data.Sex = data.Sex.replace({"M": 1, "F": 0}).infer_objects(copy=False)
        data.Ascites = data.Ascites.replace({"N": 0, "Y": 1}).infer_objects(copy=False)
        data.Hepatomegaly = data.Hepatomegaly.replace(
            {"N": 0, "Y": 1}
        ).infer_objects(copy=False)
        data.Spiders = data.Spiders.replace({"N": 0, "Y": 1}).infer_objects(copy=False)
        data.Edema = data.Edema.replace({"N": 0, "Y": 1, "S": 0.5}).infer_objects(
            copy=False
        )

    for col in ["Cholesterol", "Copper", "Tryglicerides", "Platelets"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.fillna(
        {
            "Drug": data.Drug.mode()[0],
            "Ascites": data.Ascites.mode()[0],
            "Hepatomegaly": data.Hepatomegaly.mode()[0],
            "Spiders": data.Spiders.mode()[0],
            "Cholesterol": data.Cholesterol.mean(),
            "Copper": data.Copper.mean(),
            "Alk_Phos": data.Alk_Phos.mean(),
            "SGOT": data.SGOT.mean(),
            "Tryglicerides": data.Tryglicerides.mean(),
            "Platelets": data.Platelets.mean(),
            "Prothrombin": data.Prothrombin.mean(),
            "Stage": data.Stage.mode()[0],
        }
    )
    data.reset_index(drop=True, inplace=True)

    skip_cols = [
        "Drug",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Stage",
        "event",
        "time",
    ]
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_heart_failure() -> tuple[pd.DataFrame, list[str]]:
    with open("data/heart_failure.pkl", "rb") as f:
        heart_failure = pickle.load(f)

    data = pd.concat([heart_failure.data.features, heart_failure.data.targets], axis=1)
    data = data.rename(columns={"death_event": "event"})
    cols_standardize = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
    ]
    return data, cols_standardize


def _standardize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(float)
    v = v - v.mean()
    return v / (v.std(ddof=0) + eps)


def _gram_schmidt_3(
    u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = u1.astype(float).copy()
    u2 = u2.astype(float).copy()
    u3 = u3.astype(float).copy()
    u1 -= u1.mean()
    u2 -= u2.mean()
    u3 -= u3.mean()

    h1 = u1
    n1 = float(h1 @ h1)
    if n1 < eps:
        raise ValueError("Cannot build the first semi-synthetic factor.")

    h2 = u2 - ((u2 @ h1) / n1) * h1
    n2 = float(h2 @ h2)
    if n2 < eps:
        raise ValueError("Cannot build the second semi-synthetic factor.")

    h3 = u3 - ((u3 @ h1) / n1) * h1 - ((u3 @ h2) / n2) * h2
    if float(h3 @ h3) < eps:
        raise ValueError("Cannot build the third semi-synthetic factor.")
    return h1, h2, h3


def make_semi_synth(
    dataset: str,
    seed: int = 0,
    b0e: float = 1.5,
    b1e: float = 0.8,
    b2e: float = -0.6,
    sigma_e: float = 0.5,
    b0c: float = 1.2,
    b2c: float = 0.7,
    b3c: float = 0.7,
    sigma_c: float = 0.5,
    target_censor_frac: float | None = None,
    max_bisect_iter: int = 60,
    tol: float = 1e-3,
) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    df, cols_standardize = make_survival_data(dataset)
    features = df.drop(columns=["time", "event"]).columns.tolist()
    x = df.drop(columns=["time", "event"]).to_numpy().astype(float)

    n, d = x.shape
    if n < 3:
        raise ValueError("Need at least three samples for the semi-synthetic generator.")

    h1, h2, h3 = _gram_schmidt_3(
        x @ rng.normal(size=d), np.tanh(x @ rng.normal(size=d)), (x @ rng.normal(size=d)) ** 2
    )
    h1 = _standardize(h1)
    h2 = _standardize(h2)
    h3 = _standardize(h3)

    def sample_times(b0c_local: float):
        log_te = b0e + b1e * h1 + b2e * h2 + sigma_e * rng.normal(size=n)
        log_tc = b0c_local + b2c * h2 + b3c * h3 + sigma_c * rng.normal(size=n)
        te = np.exp(log_te)
        tc = np.exp(log_tc)
        observed_time = np.minimum(te, tc)
        event = (te <= tc).astype(int)
        return te, tc, observed_time, event

    if target_censor_frac is not None:
        if not 0.0 < target_censor_frac < 1.0:
            raise ValueError("target_censor_frac must be in (0, 1).")

        def censor_frac_at(b0c_try: float) -> float:
            _, _, _, event = sample_times(b0c_try)
            return float((event == 0).mean())

        lo, hi = -10.0, 10.0
        flo = censor_frac_at(lo) - target_censor_frac
        fhi = censor_frac_at(hi) - target_censor_frac
        if flo * fhi > 0:
            lo, hi = -20.0, 20.0
            flo = censor_frac_at(lo) - target_censor_frac
            fhi = censor_frac_at(hi) - target_censor_frac
        if flo * fhi <= 0:
            for _ in range(max_bisect_iter):
                mid = 0.5 * (lo + hi)
                fmid = censor_frac_at(mid) - target_censor_frac
                if abs(fmid) < tol:
                    b0c = mid
                    break
                if flo * fmid <= 0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            else:
                b0c = 0.5 * (lo + hi)

    te, tc, observed_time, event = sample_times(b0c)
    data = pd.DataFrame(
        {
            "time": observed_time,
            "event": event,
            "true_time": te,
            "true_censor": tc,
            **{feature: df[feature].values for feature in features},
        }
    )
    data = data[data.time != 0].reset_index(drop=True)
    return data, cols_standardize
