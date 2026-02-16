"""
derive_target02_rules.py

Purpose (for report / debugging / justification):
- Learn a small piecewise-linear rule system that reproduces target02 from dataset features.
- Uses a decision tree ONLY to identify:
    (a) the switch feature (regime selector)
    (b) its split thresholds
  Then fits a linear model per region using least squares.

Outputs:
- Prints: top features, thresholds, per-region formulas, and high-precision metrics.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor


@dataclass
class RegionRule:
    # boundaries defined by switch feature thresholds
    # region i is: (t[i-1], t[i]] except first <= t0 and last > t_last
    intercept: float
    weights: Dict[str, float]  # feature_name -> weight


@dataclass
class RuleSystem:
    switch_feature: str
    thresholds: List[float]
    feature_names: List[str]  # linear model features (includes switch feature for completeness)
    regions: List[RegionRule]


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    diff = y_pred - y_true
    sse = float(np.sum(diff ** 2))
    tss = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = 1.0 - (sse / tss if tss > 0 else 0.0)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    return r2, rmse, max_abs


def fit_linear_least_squares(X: pd.DataFrame, y: pd.Series, feature_list: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Fits: y = a + sum_i b_i * x_i  using least squares.
    Returns intercept a and weights dict.
    """
    A = np.column_stack([np.ones(len(X)), X[feature_list].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(dtype=float), rcond=None)
    intercept = float(coef[0])
    weights = {feature_list[i]: float(coef[i + 1]) for i in range(len(feature_list))}
    return intercept, weights


def extract_thresholds_for_feature(tree: DecisionTreeRegressor, feature_index: int) -> List[float]:
    t = tree.tree_
    thresholds = []
    for node in range(t.node_count):
        if t.feature[node] == feature_index:
            thresholds.append(float(t.threshold[node]))
    thresholds = sorted(thresholds)
    # Remove duplicates that can appear due to floating representation
    cleaned = []
    for x in thresholds:
        if not cleaned or abs(x - cleaned[-1]) > 1e-12:
            cleaned.append(x)
    return cleaned


def assign_region_indices(switch_values: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """
    Region index = number of thresholds that switch_value is greater than.
    With 3 thresholds => regions 0..3.
    """
    idx = np.zeros(len(switch_values), dtype=int)
    for t in thresholds:
        idx += (switch_values > t).astype(int)
    return idx


def predict_rules(df: pd.DataFrame, rules: RuleSystem) -> np.ndarray:
    s = df[rules.switch_feature].to_numpy(dtype=float)
    region_idx = assign_region_indices(s, rules.thresholds)

    # Build design matrix for the region model features (for speed)
    # Each region rule can include only some features; compute via dict
    y_pred = np.zeros(len(df), dtype=float)

    for r, region_rule in enumerate(rules.regions):
        mask = (region_idx == r)
        if not np.any(mask):
            continue
        block = df.loc[mask, :]
        pred = np.full(block.shape[0], region_rule.intercept, dtype=float)
        for fname, w in region_rule.weights.items():
            pred += w * block[fname].to_numpy(dtype=float)
        y_pred[mask] = pred

    return y_pred


def pretty_round(x: float) -> float:
    # Snap tiny values to 0
    if abs(x) < 1e-12:
        return 0.0
    # Snap to "nice" decimals if extremely close
    for target in [0.05, 0.1, 0.25]:
        snapped = round(x / target) * target
        if abs(x - snapped) < 1e-10:
            return float(snapped)
    # Snap to 2 decimals if already close
    return float(x)


def print_rules(rules: RuleSystem) -> None:
    print("\n=== DISCOVERED RULE SYSTEM ===")
    print("Switch feature:", rules.switch_feature)
    print("Thresholds:", [f"{t:.16f}" for t in rules.thresholds])
    print("Linear features:", rules.feature_names)

    ts = rules.thresholds
    for i, reg in enumerate(rules.regions):
        if i == 0:
            cond = f"{rules.switch_feature} <= {ts[0]:.16f}"
        elif i < len(ts):
            cond = f"{ts[i-1]:.16f} < {rules.switch_feature} <= {ts[i]:.16f}"
        else:
            cond = f"{rules.switch_feature} > {ts[-1]:.16f}"

        parts = []
        if reg.intercept != 0.0:
            parts.append(f"{reg.intercept:+.12g}")
        for k, v in reg.weights.items():
            if v == 0.0:
                continue
            parts.append(f"{v:+.12g}*{k}")
        expr = " ".join(parts) if parts else "0.0"

        print(f"\nRegion {i}: if {cond}")
        print("  target02 =", expr)


def derive_rules(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    *,
    max_depth: int = 4,
    min_samples_leaf: int = 100,
    random_state: int = 0,
    top_k_features: int = 4,
    out_dir: str = ".",
) -> RuleSystem:
    # Train a small tree to discover regime structure
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X_fit, y_fit)

    # PLOT: Decision tree (probe)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tree_plot_path = os.path.join(out_dir, "Decision_tree.png")

    plt.figure(figsize=(22, 10))
    plot_tree(
        tree,
        feature_names=X_fit.columns.tolist(),
        filled=True,
        rounded=True,
        max_depth=4,  # matches the tree config; keeps plot readable
        fontsize=7,
    )
    plt.title("Decision Tree Probe (used to extract switch feature + thresholds)")
    plt.tight_layout()
    plt.savefig(tree_plot_path, dpi=200)
    plt.close()

    importances = pd.Series(tree.feature_importances_, index=X_fit.columns).sort_values(ascending=False)
    top_features = list(importances.head(top_k_features).index)

    switch_feature = top_features[0]
    switch_index = int(switch_feature.split("_")[1])  # columns are like feat_103 -> 103

    thresholds = extract_thresholds_for_feature(tree, switch_index)
    if len(thresholds) == 0:
        raise RuntimeError(f"No thresholds found for switch feature {switch_feature}. Increase max_depth or check tree.")

    # Region assignment
    region_idx = assign_region_indices(X_fit[switch_feature].to_numpy(dtype=float), thresholds)
    n_regions = len(thresholds) + 1

    # Fit linear model per region using least squares on the top features
    regions: List[RegionRule] = []
    for r in range(n_regions):
        mask = (region_idx == r)
        if int(mask.sum()) < 20:
            # Shouldn't happen for this dataset, but safe fallback
            intercept = float(y_fit[mask].mean()) if int(mask.sum()) > 0 else float(y_fit.mean())
            weights = {f: 0.0 for f in top_features}
        else:
            intercept, weights = fit_linear_least_squares(X_fit.loc[mask], y_fit.loc[mask], top_features)

        # Round for human-readable / report-friendly coefficients
        intercept = pretty_round(intercept)
        weights = {k: pretty_round(v) for k, v in weights.items()}
        regions.append(RegionRule(intercept=intercept, weights=weights))

    return RuleSystem(
        switch_feature=switch_feature,
        thresholds=thresholds,
        feature_names=top_features,
        regions=regions,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset_4.csv (features)")
    ap.add_argument("--target", required=True, help="Path to target_4.csv (must contain target02 column)")
    ap.add_argument("--eval", default=None, help="Optional: Path to EVAL_4.csv for generating predictions")
    ap.add_argument("--out_dir", default=".", help="Output directory for JSON + predictions")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    X = pd.read_csv(args.dataset)
    y_df = pd.read_csv(args.target)
    if "target02" not in y_df.columns:
        raise ValueError("target file must contain a 'target02' column")
    y = y_df["target02"]

    # Leak-safety (should always be true)
    assert "target02" not in X.columns, "Leakage: target02 column found inside dataset features!"

    print("Loaded X shape:", X.shape, "| y length:", len(y))

    # Full-data derivation (best rule recovery)
    rules_full = derive_rules(X, y, max_depth=4, min_samples_leaf=100, random_state=0, out_dir=out_dir)
    print_rules(rules_full)

    y_pred_full = predict_rules(X, rules_full)

    # PLOT: Distribution of true vs predicted target02
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hist_path = os.path.join(out_dir, "target02_true_vs_pred_hist.png")

    y_true = y.to_numpy(dtype=float)
    y_pred = y_pred_full.astype(float)

    # Use common bins so overlay is meaningful
    bins = 60
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    bin_edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=bin_edges, alpha=0.6, label="True target02")
    plt.hist(y_pred, bins=bin_edges, alpha=0.6, label="Predicted target02")
    plt.title("Distribution of target02: True vs Predicted")
    plt.xlabel("target02 value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    r2, rmse, max_abs = r2_rmse(y.to_numpy(dtype=float), y_pred_full)
    print("\n=== FULL-DATA METRICS (expected ~perfect for deterministic rules) ===")
    print(f"R2:       {r2:.16f}")
    print(f"RMSE:     {rmse:.3e}")
    print(f"max|err|: {max_abs:.3e}")


if __name__ == "__main__":
    main()
