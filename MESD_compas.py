"""
MESD — COMPAS Dataset (ProPublica)
====================================
Replaces the suspicious Recidivism dataset with the standard ProPublica 
COMPAS dataset, following their published preprocessing pipeline.

Protected attributes: race (African-American vs Caucasian), sex (Male/Female)
Target: two_year_recid (binary)
Intersectional groups: race × sex = 4 subgroups

Runs EXP1 ONLY (5-fold stratified CV) with all 5 methods.
Includes EDiff metric, train/val/test split for UEF, incremental saving.

Usage:
  python MESD_compas.py

  Or with custom path:
  python MESD_compas.py --path /path/to/compas-scores-two-years.csv

Output:
  results_compas/compas/exp1/exp1_raw.csv
  results_compas/compas/exp1/exp1_summary.csv
  results_compas/compas/exp1/exp1_wilcoxon.csv
"""

import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from scipy.stats import wilcoxon

import shap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# Import MESD calculator from main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MESD_unified import (MESDCalculator, compute_ediff, predict_proba_1d,
                          set_random_seeds, to_1d_groups)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

N_FOLDS = 5
MESD_SAMPLE_SIZE_OPTIM = 50
MESD_SAMPLE_SIZE_FINAL = 200
RESULTS_DIR = "results_compas"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  COMPAS DATASET LOADER                                           ║
# ║  Following ProPublica's preprocessing:                           ║
# ║  https://github.com/propublica/compas-analysis                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_compas(path=None):
    """
    Load and preprocess ProPublica COMPAS dataset.
    
    Preprocessing follows ProPublica's methodology:
    1. Filter: days_b_screening_arrest between -30 and 30
    2. Filter: is_recid != -1
    3. Filter: c_charge_degree != 'O' (ordinary traffic offenses)
    4. Filter: score_text != 'N/A'
    5. Keep only African-American and Caucasian (for binary race comparison)
    
    Protected: race (African-American=0, Caucasian=1), sex (Female=0, Male=1)
    Target: two_year_recid (0=no recidivism, 1=recidivism)
    Features: age, juv_fel_count, juv_misd_count, juv_other_count, 
              priors_count, c_charge_degree (F/M)
    """
    # Find the file
    search_paths = [
        path,
        "compas-scores-two-years.csv",
        "data/compas-scores-two-years.csv",
        "/home/n88r684/MESD/compas-scores-two-years.csv",
        os.path.expanduser("~/MESD/compas-scores-two-years.csv"),
    ]
    
    df = None
    for p in search_paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded COMPAS from {p}: {len(df)} rows")
            break
    
    if df is None:
        raise FileNotFoundError(
            "COMPAS dataset not found. Download from:\n"
            "  https://raw.githubusercontent.com/propublica/compas-analysis/"
            "master/compas-scores-two-years.csv\n"
            "Place as compas-scores-two-years.csv in the working directory.")
    
    print(f"  Raw columns: {list(df.columns)[:15]}...")
    print(f"  Raw shape: {df.shape}")
    
    # ---- ProPublica filtering ----
    # 1. days_b_screening_arrest between -30 and 30
    df = df[(df["days_b_screening_arrest"] >= -30) & 
            (df["days_b_screening_arrest"] <= 30)]
    print(f"  After screening arrest filter: {len(df)}")
    
    # 2. is_recid != -1
    df = df[df["is_recid"] != -1]
    print(f"  After is_recid filter: {len(df)}")
    
    # 3. c_charge_degree != 'O'
    df = df[df["c_charge_degree"] != "O"]
    print(f"  After charge degree filter: {len(df)}")
    
    # 4. score_text != 'N/A' (handle both string and actual NaN)
    df = df[df["score_text"].notna()]
    df = df[df["score_text"] != "N/A"]
    print(f"  After score_text filter: {len(df)}")
    
    # 5. Keep only African-American and Caucasian
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    print(f"  After race filter (AA + Caucasian only): {len(df)}")
    
    # ---- Protected attributes ----
    df["race_bin"] = (df["race"] == "Caucasian").astype(int)  # AA=0, Caucasian=1
    df["sex_bin"] = (df["sex"] == "Male").astype(int)          # Female=0, Male=1
    
    protected_cols = ["race_bin", "sex_bin"]
    
    # ---- Target ----
    target_col = "two_year_recid"
    assert target_col in df.columns, f"Missing target column '{target_col}'"
    
    # ---- Features ----
    # Following ProPublica: age, priors, juvenile counts, charge degree
    # Explicitly excluding: decile_score, score_text (these ARE the COMPAS prediction)
    # Excluding: is_recid (label leakage), name, dates, case numbers, etc.
    feature_cols = [
        "age",
        "juv_fel_count",
        "juv_misd_count", 
        "juv_other_count",
        "priors_count",
    ]
    
    # c_charge_degree: F=felony, M=misdemeanor -> binary
    df["charge_felony"] = (df["c_charge_degree"] == "F").astype(int)
    feature_cols.append("charge_felony")
    
    # Verify no leakage columns
    LEAKAGE_COLS = {"decile_score", "score_text", "v_decile_score", "v_score_text",
                    "is_recid", "is_violent_recid", "two_year_recid", "event",
                    "r_charge_degree", "r_days_from_arrest", "r_offense_date",
                    "r_charge_desc", "r_jail_in", "r_jail_out", "r_case_number",
                    "vr_case_number", "vr_charge_degree", "vr_offense_date", 
                    "vr_charge_desc", "violent_recid"}
    used_features = set(feature_cols)
    overlap = used_features & LEAKAGE_COLS
    assert len(overlap) == 0, f"LEAKAGE: features {overlap} overlap with target/score columns!"
    
    print(f"\n  Features used: {feature_cols}")
    print(f"  Leakage check: PASSED (no overlap with target/score columns)")
    
    # ---- Build arrays ----
    # Drop any rows with NaN in features or target
    cols_needed = feature_cols + protected_cols + [target_col]
    df_clean = df[cols_needed].dropna()
    
    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[target_col].values.astype(int)
    sens = df_clean[protected_cols].values.astype(int)
    
    # ---- Summary ----
    n_groups = len(np.unique([tuple(r) for r in sens]))
    print(f"\n  Final dataset: X={X.shape}, y balance={np.mean(y):.2f} (recid rate)")
    print(f"  Protected: {protected_cols}, intersectional groups={n_groups}")
    print(f"  Race: African-American={np.sum(sens[:,0]==0)}, Caucasian={np.sum(sens[:,0]==1)}")
    print(f"  Sex: Female={np.sum(sens[:,1]==0)}, Male={np.sum(sens[:,1]==1)}")
    print(f"\n  Group breakdown:")
    for r, rname in [(0, "African-American"), (1, "Caucasian")]:
        for s, sname in [(0, "Female"), (1, "Male")]:
            mask = (sens[:, 0] == r) & (sens[:, 1] == s)
            n = mask.sum()
            recid_rate = y[mask].mean() if n > 0 else 0
            print(f"    ({r},{s}) {rname:>18s} × {sname:>6s}: n={n:5d}, recid={recid_rate:.2f}")
    
    return X, y, sens, {
        "name": "compas",
        "protected_cols": protected_cols,
        "n_groups": n_groups,
        "feature_names": feature_cols,
        "uef_pop": 10, "uef_gen": 10,
        "reductions_eps": 0.02,
        "adv_lambda": 0.3, "adv_epochs": 25,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MODEL UTILITIES                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def create_model(n_features, lr=1e-3, l2_reg=1e-3, dropout=0.2):
    mdl = Sequential([
        Dense(64, activation="relu", input_dim=n_features, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(32, activation="relu", kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="binary_crossentropy", metrics=["accuracy"])
    return mdl


def _safe_mask_eq(arr, val):
    return np.array([a == val for a in arr])


def dp_intersectional_maxgap(y, yhat, sensitive_features):
    sens = to_1d_groups(sensitive_features)
    rates = {}
    for g in np.unique(sens):
        m = _safe_mask_eq(sens, g)
        if m.sum() > 0:
            rates[g] = float(np.mean(yhat[m]))
    if len(rates) < 2:
        return 0.0
    vals = list(rates.values())
    return float(max(vals) - min(vals))


def eo_intersectional_maxgap(y, yhat, sensitive_features):
    sens = to_1d_groups(sensitive_features)
    tpr = {}
    for g in np.unique(sens):
        m = _safe_mask_eq(sens, g) & (y == 1)
        if m.sum() > 0:
            tpr[g] = float(np.mean(yhat[m]))
    if len(tpr) < 2:
        return 0.0
    vals = list(tpr.values())
    return float(max(vals) - min(vals))


def to_fairlearn_labels(sens):
    g = to_1d_groups(sens)
    return np.array([str(x) for x in g])


def compute_reweighing_weights(y, sens):
    g = to_1d_groups(sens)
    w = np.ones(len(y), float)
    N = len(y)
    for gi in np.unique(g):
        for yi in np.unique(y):
            m = _safe_mask_eq(g, gi) & (y == yi)
            expected = (np.sum(_safe_mask_eq(g, gi)) * np.sum(y == yi)) / N
            actual = np.sum(m)
            if actual > 0:
                w[m] = expected / actual
    return w


def wrap_sklearn(est):
    class _W:
        def __init__(self, e):
            self._e = e
        def predict(self, X, **kwargs):
            return predict_proba_1d(self._e, X).reshape(-1, 1)
        def predict_proba(self, X):
            if hasattr(self._e, "predict_proba"):
                try:
                    return self._e.predict_proba(X)
                except Exception:
                    pass
            if hasattr(self._e, "_pmf_predict"):
                try:
                    return self._e._pmf_predict(X)
                except Exception:
                    pass
            p = self._predict_raw(X)
            return np.column_stack([1 - p, p])
        def _predict_raw(self, X):
            if hasattr(self._e, "predict"):
                try:
                    pred = self._e.predict(X)
                except TypeError:
                    pred = self._e.predict(X)
                pred = np.asarray(pred, dtype=float).ravel()
                return np.clip(pred, 0, 1)
            return np.zeros(X.shape[0])
    return _W(est)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  BASELINE RUNNERS                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_erm(X_tr, y_tr, seed=42):
    set_random_seeds(seed)
    mdl = create_model(X_tr.shape[1])
    mdl.fit(X_tr, y_tr, epochs=30, batch_size=32, verbose=0)
    return mdl


def run_reweighing(X_tr, y_tr, sens_tr, seed=42):
    set_random_seeds(seed)
    w = compute_reweighing_weights(y_tr, sens_tr)
    mdl = create_model(X_tr.shape[1])
    mdl.fit(X_tr, y_tr, sample_weight=w, epochs=30, batch_size=32, verbose=0)
    return mdl


def run_reductions(X_tr, y_tr, sens_tr, eps=0.02, seed=42):
    set_random_seeds(seed)
    base = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    eg = ExponentiatedGradient(base, DemographicParity(), eps=float(eps))
    eg.fit(X_tr, y_tr, sensitive_features=to_fairlearn_labels(sens_tr))
    return eg


def run_adversarial(X_tr, y_tr, sens_tr, seed=42, lambd=0.3, epochs=25):
    """Adversarial debiasing via group-aware sample reweighting."""
    set_random_seeds(seed)
    sens_1d = to_1d_groups(sens_tr)
    unique_g = np.unique(sens_1d)
    n_groups = len(unique_g)
    
    mdl = create_model(X_tr.shape[1])
    mdl.fit(X_tr, y_tr, epochs=max(1, epochs // 2), batch_size=32, verbose=0)
    
    # Adversarial-inspired reweighting
    p = predict_proba_1d(mdl, X_tr)
    yhat = (p >= 0.5).astype(int)
    
    weights = np.ones(len(y_tr), dtype=float)
    for gi in unique_g:
        mask = _safe_mask_eq(sens_1d, gi)
        group_size = mask.sum()
        if group_size == 0:
            continue
        group_weight = len(y_tr) / (n_groups * group_size)
        wrong = mask & (yhat != y_tr)
        weights[mask] = group_weight
        weights[wrong] = group_weight * (1 + lambd)
    
    weights = weights / weights.mean()
    mdl.fit(X_tr, y_tr, sample_weight=weights, 
            epochs=max(1, epochs // 2), batch_size=32, verbose=0)
    return mdl


# ╔══════════════════════════════════════════════════════════════════╗
# ║  EVALUATION + UEF                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def evaluate_model(model, X_test, y_test, sens_test, mesd_calc, X_bg,
                   threshold=0.5, seed=42, sample_size=None):
    if sample_size is None:
        sample_size = MESD_SAMPLE_SIZE_FINAL
    set_random_seeds(seed)
    p = predict_proba_1d(model, X_test)
    yhat = (p >= threshold).astype(int)
    auc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else 0.5
    f1 = f1_score(y_test, yhat) if len(np.unique(y_test)) > 1 else 0.0
    dp = dp_intersectional_maxgap(y_test, yhat, sens_test)
    eo = eo_intersectional_maxgap(y_test, yhat, sens_test)
    mesd = mesd_calc.calculate_mesd(model, X_test, y_test, sens_test, X_bg, sample_size)
    ediff_val = compute_ediff(model, X_test, sens_test, X_bg,
                              sample_size=min(sample_size, 100), seed=seed)
    return {
        "auc": float(auc), "f1": float(f1), "dp": float(dp), "eo": float(eo),
        "mesd_cvar": float(mesd["mesd_cvar"]), "mesd_max": float(mesd["mesd_max"]),
        "mesd_var": float(mesd["mesd_var"]), "ediff": float(ediff_val),
        "worst_group_stability": mesd.get("worst_group_stability", np.nan),
    }


# --- NSGA-II ---
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination


def chebyshev_select(F, ideal, nadir, weights=None):
    F, ideal, nadir = [np.asarray(a, float) for a in [F, ideal, nadir]]
    Z = (F - ideal) / np.maximum(nadir - ideal, 1e-12)
    w = np.ones(F.shape[1]) / F.shape[1] if weights is None else np.asarray(weights, float)
    return int(np.argmin(np.max(w * Z, axis=1)))


class NNHyperparamProblem(Problem):
    """NSGA-II problem — evaluates on VALIDATION set (not test)."""
    def __init__(self, X_tr, y_tr, X_val, y_val, sens_val, mesd_calc, seed=42):
        super().__init__(n_var=5, n_obj=3, n_constr=0,
                         xl=np.array([1e-5, 1e-4, 0.0, 10, 0.2]),
                         xu=np.array([5e-3, 5e-2, 0.5, 60, 0.8]))
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.sens_val, self.mesd_calc, self.seed = sens_val, mesd_calc, seed
        self._eval_count = 0
        self._t0 = time.time()

    def _evaluate(self, pop, out, *args, **kwargs):
        F = []
        for row in np.atleast_2d(pop):
            lr, l2r, dr, ep, th = row
            mdl = create_model(self.X_tr.shape[1], float(lr), float(l2r), float(dr))
            mdl.fit(self.X_tr, self.y_tr, epochs=int(round(ep)), batch_size=32, verbose=0)
            m = evaluate_model(mdl, self.X_val, self.y_val, self.sens_val,
                               self.mesd_calc, self.X_tr, float(th), self.seed,
                               sample_size=MESD_SAMPLE_SIZE_OPTIM)
            F.append([1 - m["auc"], m["dp"], m["mesd_cvar"]])
            self._eval_count += 1
            if self._eval_count % 10 == 0:
                elapsed = time.time() - self._t0
                print(f"    [NSGA-II] {self._eval_count} evals, "
                      f"{elapsed/self._eval_count:.1f}s/eval, {elapsed/60:.1f}min")
        out["F"] = np.asarray(F, float)


def run_uef_nsga2(X_tr, y_tr, X_val, y_val, sens_val, mesd_calc,
                  pop_size=10, ngen=10, seed=42):
    total = pop_size * ngen
    print(f"    [UEF] Starting NSGA-II: pop={pop_size}, gen={ngen}, evals={total}")
    prob = NNHyperparamProblem(X_tr, y_tr, X_val, y_val, sens_val, mesd_calc, seed)
    t0 = time.time()
    res = pymoo_minimize(prob, NSGA2(pop_size=pop_size),
                         get_termination("n_gen", ngen), seed=seed, verbose=False)
    wt = time.time() - t0
    print(f"    [UEF] Complete: {wt/60:.1f} min, {prob._eval_count} evals")
    F, Xp = res.F, res.X
    k = chebyshev_select(F, np.min(F, 0), np.max(F, 0))
    sel = Xp[k]
    return {"lr": float(sel[0]), "l2": float(sel[1]), "dropout": float(sel[2]),
            "epochs": int(round(sel[3])), "threshold": float(sel[4])}


# ╔══════════════════════════════════════════════════════════════════╗
# ║  EXP1 — 5-Fold Stratified CV                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_all_methods_one_fold(X_tr, y_tr, X_te, y_te, s_tr, s_te, mc, cfg, fold_id):
    R = {}
    seed = 42

    print(f"  [fold={fold_id}] ERM...")
    t0 = time.time()
    R["ERM"] = evaluate_model(run_erm(X_tr, y_tr, seed), X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["ERM"]["wall_clock_s"] = time.time() - t0

    print(f"  [fold={fold_id}] UEF...")
    t0 = time.time()
    X_tr_in, X_val, y_tr_in, y_val, s_tr_in, s_val = train_test_split(
        X_tr, y_tr, s_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    best_p = run_uef_nsga2(X_tr_in, y_tr_in, X_val, y_val, s_val, mc,
                           cfg.get("uef_pop", 10), cfg.get("uef_gen", 10), seed)
    mdl_final = create_model(X_tr.shape[1], best_p["lr"], best_p["l2"], best_p["dropout"])
    mdl_final.fit(X_tr, y_tr, epochs=best_p["epochs"], batch_size=32, verbose=0)
    R["UEF"] = evaluate_model(mdl_final, X_te, y_te, s_te, mc, X_tr,
                              best_p["threshold"], seed, sample_size=MESD_SAMPLE_SIZE_FINAL)
    R["UEF"]["wall_clock_s"] = time.time() - t0

    print(f"  [fold={fold_id}] RW...")
    t0 = time.time()
    R["RW"] = evaluate_model(run_reweighing(X_tr, y_tr, s_tr, seed), X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["RW"]["wall_clock_s"] = time.time() - t0

    print(f"  [fold={fold_id}] RD...")
    t0 = time.time()
    rd = run_reductions(X_tr, y_tr, s_tr, cfg.get("reductions_eps", 0.02), seed)
    R["RD"] = evaluate_model(wrap_sklearn(rd), X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["RD"]["wall_clock_s"] = time.time() - t0

    print(f"  [fold={fold_id}] ADL...")
    t0 = time.time()
    R["ADL"] = evaluate_model(
        run_adversarial(X_tr, y_tr, s_tr, seed, cfg.get("adv_lambda", 0.3), cfg.get("adv_epochs", 25)),
        X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["ADL"]["wall_clock_s"] = time.time() - t0

    return R


def run_exp1(X, y, sens, cfg, outdir):
    dataset_name = cfg["name"]
    print("\n" + "=" * 70)
    print(f"EXP1: {dataset_name} ({N_FOLDS}-Fold Stratified CV)")
    print("=" * 70)
    os.makedirs(outdir, exist_ok=True)

    METRICS = ["auc", "f1", "dp", "eo", "mesd_cvar", "mesd_max", "mesd_var", "ediff"]

    # Resume support
    raw_path = os.path.join(outdir, "exp1_raw.csv")
    if os.path.exists(raw_path):
        existing = pd.read_csv(raw_path)
        completed_folds = set(existing["fold"].unique())
        all_rows = existing.to_dict("records")
        print(f"  Resuming: {len(completed_folds)} completed folds")
    else:
        completed_folds = set()
        all_rows = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if fold_id in completed_folds:
            print(f"\n--- Fold {fold_id} --- SKIPPED (already complete)")
            continue

        print(f"\n--- Fold {fold_id}/{N_FOLDS-1} ---")
        fold_t0 = time.time()

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        s_tr, s_te = sens[train_idx], sens[test_idx]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        mc = MESDCalculator(cvar_alpha=0.1, random_seed=42)

        R = run_all_methods_one_fold(X_tr, y_tr, X_te, y_te, s_tr, s_te, mc, cfg, fold_id)

        for mname, mdict in R.items():
            row = {"dataset": dataset_name, "method": mname, "fold": fold_id}
            for m in METRICS:
                row[m] = mdict.get(m, np.nan)
            row["wall_clock_s"] = mdict.get("wall_clock_s", np.nan)
            row["worst_group_stability"] = mdict.get("worst_group_stability", np.nan)
            all_rows.append(row)

        # INCREMENTAL SAVE
        raw = pd.DataFrame(all_rows)
        raw.to_csv(raw_path, index=False)
        fold_elapsed = (time.time() - fold_t0) / 60
        print(f"  Fold {fold_id} complete in {fold_elapsed:.1f} min — saved to {raw_path}")

    # Final aggregation
    raw = pd.DataFrame(all_rows)
    summary = []
    for meth in ["ERM", "RW", "RD", "ADL", "UEF"]:
        sub = raw[raw["method"] == meth]
        r = {"dataset": dataset_name, "method": meth}
        for m in METRICS + ["wall_clock_s", "worst_group_stability"]:
            v = sub[m].dropna().values
            r[f"{m}_mean"] = float(np.mean(v)) if len(v) else np.nan
            r[f"{m}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary.append(r)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(outdir, "exp1_summary.csv"), index=False)

    # Wilcoxon
    uef_key = "UEF"
    if uef_key in raw["method"].values:
        wrows = []
        udf = raw[raw["method"] == uef_key].set_index("fold")
        for meth in raw["method"].unique():
            if meth == uef_key:
                continue
            odf = raw[raw["method"] == meth].set_index("fold")
            common = sorted(set(udf.index) & set(odf.index))
            for m in METRICS:
                if len(common) < 5:
                    wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                  "p": np.nan, "note": f"n={len(common)}"})
                    continue
                u = udf.loc[common, m].values.astype(float)
                o = odf.loc[common, m].values.astype(float)
                if np.all(u == o):
                    wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                  "p": 1.0, "note": "identical"})
                else:
                    try:
                        s, p_val = wilcoxon(u, o)
                        wrows.append({"vs": meth, "metric": m, "stat": float(s),
                                      "p": float(p_val), "note": ""})
                    except Exception as e:
                        wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                      "p": np.nan, "note": str(e)})
        pd.DataFrame(wrows).to_csv(os.path.join(outdir, "exp1_wilcoxon.csv"), index=False)

    # Print
    print(f"\n{'='*70}")
    print(f"EXP1 SUMMARY — {dataset_name} ({N_FOLDS}-fold CV)")
    print(f"{'='*70}")
    for _, r in sdf.iterrows():
        print(f"  {r['method']:5s}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
              f"F1={r['f1_mean']:.3f}±{r['f1_std']:.3f}  "
              f"DP={r['dp_mean']:.3f}±{r['dp_std']:.3f}  "
              f"EOD={r['eo_mean']:.3f}±{r['eo_std']:.3f}  "
              f"MESD={r['mesd_cvar_mean']:.4f}±{r['mesd_cvar_std']:.4f}  "
              f"EDiff={r['ediff_mean']:.4f}±{r['ediff_std']:.4f}")
    
    return raw, sdf


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="MESD COMPAS Experiment")
    parser.add_argument("--path", default=None, help="Path to compas-scores-two-years.csv")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MESD — COMPAS Dataset (ProPublica)")
    print("Replacing Recidivism dataset to address data leakage concern")
    print("=" * 70)
    
    t0 = time.time()
    
    X, y, sens, cfg = load_compas(args.path)
    outdir = os.path.join(RESULTS_DIR, "compas", "exp1")
    run_exp1(X, y, sens, cfg, outdir)
    
    elapsed = (time.time() - t0) / 3600
    print(f"\n{'='*70}")
    print(f"COMPAS COMPLETE — {elapsed:.1f} hr total")
    print(f"Results in: {outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
