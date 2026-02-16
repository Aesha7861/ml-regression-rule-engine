"""Part 1 — target01 regression using CatBoost.

Pipeline overview
-----------------
- Load: dataset_{ID}.csv (X), target_{ID}.csv (y=target01), EVAL_{ID}.csv (X_eval)
- Validate: NaNs, +/-inf, numeric-only, train/eval column match
- Diagnose: global feature stats and simple train-vs-eval drift summary
- Split labeled data: TRAIN/VAL/TEST = 70/15/15
- Tune: search a small set of hyperparameter candidates using VAL only
  (TEST is untouched until the end)
- Evaluate: report RMSE + R2 on TRAIN/VAL/TEST
- Train final model: (default) train on TRAIN+VAL using the chosen iterations
- Predict eval: write EVAL_target01_{ID}.csv
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool


# 1) CONFIG
ID = 4
DATA_DIR = "problem_4"

TRAIN_X_PATH = os.path.join(DATA_DIR, f"dataset_{ID}.csv")
TRAIN_Y_PATH = os.path.join(DATA_DIR, f"target_{ID}.csv")
EVAL_X_PATH = os.path.join(DATA_DIR, f"EVAL_{ID}.csv")
OUT_PATH = os.path.join(DATA_DIR, f"EVAL_target01_{ID}.csv")

RANDOM_STATE = 42

# Toggle tuning. When False, FIXED_PARAMS will be used (fast re-runs).
RUN_TUNING = True

# Plotting
MAKE_PLOTS = True
PLOTS_DIR = "."  

# (40 is a reasonable balance for runtime vs improvements.)
MAX_CANDIDATES = 40

# 2) MODEL PARAMS
# Base params are always used as the starting point.
# The tuning loop only changes a few knobs.
BASE_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,            # cap; early stopping decides
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 30.0,
    "bootstrap_type": "Bernoulli",
    "sampling_frequency": "PerTree",
    "subsample": 0.8,
    "rsm": 0.8,
    "border_count": 128,
    "random_seed": RANDOM_STATE,
    "od_type": "Iter",
    "od_wait": 150,
    "thread_count": -1,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}

# If RUN_TUNING=False, these will be used.
FIXED_PARAMS = {
    **BASE_PARAMS,
    "depth": 7,
    "learning_rate": 0.02,
    "l2_leaf_reg": 30.0,
    "subsample": 1.0,
    "rsm": 0.8,
    "iterations": 3000,
}

# 3) IO HELPERS

def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV with stable defaults"""
    return pd.read_csv(path, low_memory=False)


# 4) SANITY CHECKS

def assert_clean_numeric(df: pd.DataFrame, name: str) -> None:
    """Ensure the dataframe contains only numeric values and no NaN/inf."""
    print(f"\nDATA CHECK: {name}")
    print("shape:", df.shape)

    total_nan = int(df.isna().sum().sum())
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    num = df.select_dtypes(include=["number"])
    total_inf = int(np.isinf(num.to_numpy()).sum()) if num.shape[1] > 0 else 0

    print("total NaNs:", total_nan)
    print("non-numeric columns:", len(non_numeric_cols))
    print("total +/-inf in numeric data:", total_inf)

    if total_nan > 0 or total_inf > 0 or len(non_numeric_cols) > 0:
        raise ValueError(f"{name} is not clean. Fix NaNs/infs/non-numeric columns first.")


def check_column_match(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> None:
    """Ensure train/eval feature columns match exactly (same set and order)."""
    print("train feature columns:", X_train.shape[1])
    print("eval feature columns :", X_eval.shape[1])
    if list(X_train.columns) != list(X_eval.columns):
        raise ValueError("Train/EVAL columns mismatch (set or order).")


def load_target01(y_df: pd.DataFrame) -> np.ndarray:
    """Load target01 as a float numpy array and validate it."""
    if "target01" not in y_df.columns:
        raise ValueError("target01 not found in target CSV.")

    y = pd.to_numeric(y_df["target01"], errors="raise")

    nan_y = int(pd.isna(y).sum())
    inf_y = int(np.isinf(y.to_numpy()).sum())
    print("target01 NaNs:", nan_y)
    print("target01 +/-inf:", inf_y)
    if nan_y > 0 or inf_y > 0:
        raise ValueError("target01 contains NaNs or infs.")

    y = y.astype(float).to_numpy()
    print("target01 mean:", float(np.mean(y)))
    print("target01 std :", float(np.std(y)))
    print("target01 min/max:", float(np.min(y)), float(np.max(y)))
    return y


# 5) DIAGNOSTICS (feature drift + plots)

def summarize_features(df: pd.DataFrame, name: str) -> None:
    """Print a compact global summary across all feature values."""
    arr = df.to_numpy(dtype=float)
    print(f"\nFEATURE SUMMARY: {name} ")
    print("shape:", df.shape)
    print("global mean:", float(arr.mean()))
    print("global std :", float(arr.std()))
    print("global min/max:", float(arr.min()), float(arr.max()))


def compute_feature_drift(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> pd.Series:
    """Compute per-feature standardized mean shift: |mu_train - mu_eval| / sigma_train."""
    train_mean = X_train.mean(axis=0)
    eval_mean = X_eval.mean(axis=0)
    train_std = X_train.std(axis=0).replace(0, np.nan)
    drift = ((train_mean - eval_mean).abs() / train_std).replace([np.inf], np.nan)
    return drift


# 6) TRAIN / VAL / TEST + TUNING

def fit_and_score(
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
):
    """Train on TRAIN, early-stop on VAL, return (rmse, r2, best_iter, model)."""
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(X_tr, y_tr),
        eval_set=Pool(X_va, y_va),
        use_best_model=True,
    )

    pred_va = model.predict(X_va)
    rmse_va = float(np.sqrt(mean_squared_error(y_va, pred_va)))
    r2_va = float(r2_score(y_va, pred_va))
    best_it = model.get_best_iteration()
    return rmse_va, r2_va, best_it, model


def build_candidates(base_params: dict) -> list[dict]:
    """Build a candidate list for a lightweight random search."""
    base = deepcopy(base_params)
    candidates: list[dict] = []

    for depth in [7, 8, 9]:
        for lr in [0.02, 0.03, 0.05]:
            for l2 in [3.0, 10.0, 30.0, 100.0]:
                for subsample in [0.6, 0.8, 1.0]:
                    for rsm in [0.6, 0.8, 1.0]:
                        p = deepcopy(base)
                        p["depth"] = depth
                        p["learning_rate"] = lr
                        p["l2_leaf_reg"] = l2
                        p["subsample"] = subsample
                        p["rsm"] = rsm
                        candidates.append(p)

    # Shuffle and truncate to a small budget
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(candidates)
    return candidates[:MAX_CANDIDATES]


def print_metrics_block(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{label:<5} RMSE: {rmse}")
    print(f"{label:<5} R2  : {r2}")


def main() -> None:
    # Load
    X = read_csv(TRAIN_X_PATH)
    y_df = read_csv(TRAIN_Y_PATH)
    X_eval = read_csv(EVAL_X_PATH)

    # Checks
    assert_clean_numeric(X, "TRAIN FEATURES (dataset)")
    assert_clean_numeric(X_eval, "EVAL FEATURES (EVAL)")
    check_column_match(X, X_eval)
    if len(X) != len(y_df):
        raise ValueError("Row mismatch between dataset and target.")
    y = load_target01(y_df)

    # Drift + optional plots
    summarize_features(X, "TRAIN FEATURES (dataset)")
    summarize_features(X_eval, "EVAL FEATURES (eval)")

    drift = compute_feature_drift(X, X_eval)
    print("\nTRAIN vs EVAL FEATURE DRIFT (summary) ")
    print("mean(|mean_train-mean_eval|/std_train):", float(np.nanmean(drift)))
    print("median(|...|):", float(np.nanmedian(drift)))
    print("max(|...|):", float(np.nanmax(drift)))


    print("\nAll checks passed. Training model...")

    # Split labeled data
    idx = np.arange(len(X))

    # 70% train, 30% temp
    X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
        X, y, idx,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # temp -> 15% val, 15% test
    X_va, X_te, y_va, y_te, idx_va, idx_te = train_test_split(
        X_tmp, y_tmp, idx_tmp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Tune on VAL only
    if RUN_TUNING:
        candidates = build_candidates(BASE_PARAMS)
        best = None  # (val_rmse, val_r2, best_iter, params, model)

        print("\nTUNING ON VALIDATION ONLY ")
        for i, p in enumerate(candidates, 1):
            val_rmse, val_r2, best_it, m = fit_and_score(p, X_tr, y_tr, X_va, y_va)
            print(
                f"{i:02d}/{len(candidates)} "
                f"depth={p['depth']} lr={p['learning_rate']} l2={p['l2_leaf_reg']} "
                f"subsample={p['subsample']} rsm={p['rsm']} "
                f"-> VAL RMSE={val_rmse:.6f} VAL R2={val_r2:.6f} best_iter={best_it}"
            )

            if best is None or val_rmse < best[0]:
                best = (val_rmse, val_r2, best_it, deepcopy(p), m)

        assert best is not None
        best_val_rmse, best_val_r2, best_iter, best_params, best_model = best

        print("\nBEST PARAMS (chosen by VAL RMSE) ")
        print("Best VAL RMSE:", best_val_rmse)
        print("Best VAL R2  :", best_val_r2)
        print(
            "Best params  :",
            {
                k: best_params[k]
                for k in ["depth", "learning_rate", "l2_leaf_reg", "subsample", "rsm"]
            },
        )
        print("Iterations cap:", best_params["iterations"])
        print("Chosen iterations (best_iter+1):", (best_iter + 1) if best_iter is not None else None)
    else:
        # No tuning: just train once using FIXED_PARAMS
        best_val_rmse, best_val_r2, best_iter, best_model = fit_and_score(
            FIXED_PARAMS, X_tr, y_tr, X_va, y_va
        )
        best_params = deepcopy(FIXED_PARAMS)
        print("\nTUNING DISABLED (using FIXED_PARAMS) ")
        print("VAL RMSE:", best_val_rmse)
        print("VAL R2  :", best_val_r2)
        print("best_iter:", best_iter)

    # Final evaluation on TRAIN/VAL/TEST (single pass)
    pred_tr = best_model.predict(X_tr)
    pred_va = best_model.predict(X_va)
    pred_te = best_model.predict(X_te)

    print("\nTRAIN / VAL / TEST METRICS (BEST) ")
    print_metrics_block("TRAIN", y_tr, pred_tr)
    print_metrics_block("VAL", y_va, pred_va)
    print_metrics_block("TEST", y_te, pred_te)

    print("\nTEST SAMPLE")
    print("First 10 true y :", y_te[:10])
    print("First 10 pred y :", pred_te[:10])
    print("First 10 abs err:", np.abs(y_te[:10] - pred_te[:10]))

    # Learning Curve (RMSE vs iteration)
    evals = best_model.get_evals_result()

    # CatBoost usually uses keys: "learn" and something like "validation" / "validation_0"
    learn_rmse = evals["learn"]["RMSE"]

    val_key = None
    for k in evals.keys():
        if k != "learn":
            val_key = k
            break
    val_rmse = evals[val_key]["RMSE"]

    iters = np.arange(1, len(learn_rmse) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iters, learn_rmse, label="Train RMSE")
    plt.plot(iters, val_rmse, label="Val RMSE")
    plt.title("Learning Curve (RMSE vs iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
 
    plt.savefig(f"learning_curve_target01_{ID}.png", dpi=200)
    plt.show()

    # Train final model 
    final_params = deepcopy(best_params)

    # Use the best iteration count discovered on VAL
    if best_iter is not None and best_iter >= 0:
        final_params["iterations"] = int(best_iter) + 1

    # Train on TRAIN+VAL 
    X_trva = pd.concat([X_tr, X_va], axis=0)
    y_trva = np.concatenate([y_tr, y_va])

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(Pool(X_trva, y_trva), use_best_model=False)

    pred_eval = final_model.predict(X_eval)
    submission = pd.DataFrame({"target01": pred_eval})
    submission.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Submission head:")
    print(submission.head())

    print("\nEVAL PREDICTION SANITY CHECK ")
    print("pred_eval mean/std:", float(np.mean(pred_eval)), float(np.std(pred_eval)))
    print("pred_eval min/max :", float(np.min(pred_eval)), float(np.max(pred_eval)))


if __name__ == "__main__":
    main()
