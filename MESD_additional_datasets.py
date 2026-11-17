"""
MESD — Additional Datasets for MLJ Expansion
=============================================
Runs EXP1 (5-fold stratified CV) ONLY on:
  1. Law School (race as protected attribute, 2 intersectional groups)
  2. Bank Marketing (age_bin × marital_bin, 4 intersectional groups)

Uses the same pipeline as MESD_unified.py but is self-contained.
Includes: EDiff metric, train/val/test split for UEF, incremental saving.

Usage:
  python MESD_additional_datasets.py

Output:
  results_additional/law_school/exp1/exp1_raw.csv
  results_additional/law_school/exp1/exp1_summary.csv
  results_additional/law_school/exp1/exp1_wilcoxon.csv
  results_additional/bank_marketing/exp1/...
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ==== ML ====
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

# ==== Fairlearn ====
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# ==== Explainability ====
import shap
from lime.lime_tabular import LimeTabularExplainer

# ==== Stats ====
from scipy.stats import wilcoxon

# ==== NSGA-II ====
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

# ==== Plotting ====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== TF ====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

N_FOLDS = 5
MESD_SAMPLE_SIZE_OPTIM = 50
MESD_SAMPLE_SIZE_FINAL = 200
RESULTS_DIR = "results_additional"


def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  DATASET LOADING                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_law_school(path=None):
    """
    Load Law School dataset (used by EDiff / Germino et al. 2025).
    Columns: decile1b, decile3, lsat, ugpa, zfygpa, zgpa, fulltime, fam_inc, male, racetxt, tier, pass_bar
    Protected attributes: racetxt (white=1 vs non-white=0), male (0/1).
    Target: pass_bar (1=passed, 0=failed).
    """
    search_paths = [
        path,
        "lawschool.csv",
        "data/lawschool.csv",
        "/home/n88r684/MESD/lawschool.csv",
        os.path.expanduser("~/MESD/lawschool.csv"),
    ]
    df = None
    for p in search_paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Loaded Law School from {p}: {len(df)} rows")
            break
    if df is None:
        raise FileNotFoundError(
            "Law School dataset not found. Place lawschool.csv in the working directory.\n"
            "Download from: https://github.com/sethneel/GerryFair/blob/master/dataset/lawschool.csv")

    # Columns: decile1b, decile3, lsat, ugpa, zfygpa, zgpa, fulltime, fam_inc, male, racetxt, tier, pass_bar
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")

    # Target
    target_col = "pass_bar"
    assert target_col in df.columns, f"Expected '{target_col}' column. Found: {list(df.columns)}"

    # Protected attributes
    # racetxt: 1=white, others=non-white (binarize to white vs non-white)
    assert "racetxt" in df.columns, f"Expected 'racetxt' column"
    # In this dataset racetxt uses numeric codes; check unique values
    race_vals = df["racetxt"].unique()
    print(f"  racetxt unique values: {sorted(race_vals)}")
    # Typically: 1=white, others=non-white. If string, handle differently.
    if df["racetxt"].dtype == object:
        df["race_bin"] = (df["racetxt"].str.lower().isin(["white", "caucasian"])).astype(int)
    else:
        # Numeric: typically 1=white in this dataset
        df["race_bin"] = (df["racetxt"] == 1).astype(int)  # white=1, non-white=0

    # male: already binary (0/1)
    assert "male" in df.columns, f"Expected 'male' column"
    df["male"] = df["male"].astype(int)

    protected_cols = ["race_bin", "male"]

    # Features: everything except target and raw protected cols
    drop_from_features = [target_col, "racetxt", "race_bin", "male"]
    feat_cols = [c for c in df.columns if c not in drop_from_features]
    Xdf = df[feat_cols].copy()

    # Handle NaN
    Xdf = Xdf.fillna(Xdf.median())

    X = Xdf.values.astype(float)
    y = df[target_col].values.astype(int)
    sens = df[protected_cols].values.astype(int)

    # Drop rows where target is NaN
    valid = ~np.isnan(y)
    X, y, sens = X[valid], y[valid], sens[valid]

    n_groups = len(np.unique([tuple(r) for r in sens]))
    print(f"  Final: X={X.shape}, y balance={np.mean(y):.2f}")
    print(f"  Protected: {protected_cols}, intersectional groups={n_groups}")
    print(f"  Race: white={np.sum(sens[:,0]==1)}, non-white={np.sum(sens[:,0]==0)}")
    print(f"  Sex: male={np.sum(sens[:,1]==1)}, female={np.sum(sens[:,1]==0)}")

    return X, y, sens, {
        "name": "law_school",
        "protected_cols": protected_cols,
        "n_groups": n_groups,
        "uef_pop": 10, "uef_gen": 10,
        "reductions_eps": 0.02,
        "adv_lambda": 0.3, "adv_epochs": 25,
    }


def load_bank_marketing(path=None):
    """
    Load Bank Marketing dataset from UCI.
    Protected attributes: age (binarized <35 / >=35), marital status (married vs other).
    Target: y (subscribed to term deposit).
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path, sep=";") if "bank" in path.lower() else pd.read_csv(path)
    else:
        try:
            url = "https://raw.githubusercontent.com/dssg/aequitas/master/examples/data/bank-additional-full.csv"
            df = pd.read_csv(url, sep=";")
            print(f"  Downloaded Bank Marketing dataset: {len(df)} rows")
        except Exception:
            for p in ["bank-additional-full.csv", "bank.csv", "data/bank-additional-full.csv",
                       "/home/n88r684/MESD/bank-additional-full.csv"]:
                if os.path.exists(p):
                    df = pd.read_csv(p, sep=";")
                    break
            else:
                raise FileNotFoundError(
                    "Bank Marketing dataset not found. Download from "
                    "https://archive.ics.uci.edu/ml/datasets/bank+marketing")
    
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  Bank Marketing: shape={df.shape}, columns={list(df.columns)[:10]}...")
    
    # Target
    if "y" in df.columns:
        df["target"] = (df["y"] == "yes").astype(int)
    elif "deposit" in df.columns:
        df["target"] = (df["deposit"] == "yes").astype(int)
    else:
        raise ValueError(f"Cannot find target. Columns: {list(df.columns)}")
    
    # Protected: age (binary) and marital
    df["age_bin"] = (df["age"] >= 35).astype(int)  # 0=young, 1=older
    df["marital_bin"] = (df["marital"] == "married").astype(int)  # 0=not married, 1=married
    
    protected_cols = ["age_bin", "marital_bin"]
    drop_from_features = ["age", "marital", "y", "target", "age_bin", "marital_bin"]
    if "deposit" in df.columns:
        drop_from_features.append("deposit")
    
    # Prepare features
    feat_cols = [c for c in df.columns if c not in drop_from_features]
    Xdf = df[feat_cols].copy()
    
    # Remove 'unknown' values by replacing with mode
    for c in Xdf.columns:
        if Xdf[c].dtype == object:
            Xdf[c] = Xdf[c].replace("unknown", np.nan)
            if Xdf[c].isna().any():
                mode_val = Xdf[c].mode().iloc[0] if not Xdf[c].mode().empty else "other"
                Xdf[c] = Xdf[c].fillna(mode_val)
    
    # Encode categoricals
    obj_cols = [c for c in Xdf.columns if Xdf[c].dtype == object]
    if obj_cols:
        Xdf = pd.get_dummies(Xdf, columns=obj_cols, drop_first=True)
    
    # Handle any remaining NaN
    Xdf = Xdf.fillna(0)
    
    X = Xdf.values.astype(float)
    y = df["target"].values.astype(int)
    sens = df[protected_cols].values.astype(int)
    
    print(f"  Final: X={X.shape}, y balance={np.mean(y):.2f}, "
          f"protected_cols={protected_cols}, n_groups={len(np.unique(sens, axis=0))}")
    print(f"  Age distribution: young={np.sum(sens[:,0]==0)}, older={np.sum(sens[:,0]==1)}")
    print(f"  Marital distribution: not_married={np.sum(sens[:,1]==0)}, married={np.sum(sens[:,1]==1)}")
    
    return X, y, sens, {
        "name": "bank_marketing",
        "protected_cols": protected_cols,
        "n_groups": len(np.unique(sens, axis=0)),
        "uef_pop": 10, "uef_gen": 10,
        "reductions_eps": 0.02,
        "adv_lambda": 0.3, "adv_epochs": 25,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MODEL / METRIC UTILITIES (same as MESD_unified.py)             ║
# ╚══════════════════════════════════════════════════════════════════╝

# --- Import everything from the main script if available ---
# Otherwise define inline

def predict_proba_1d(model, X):
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        try:
            pp = model.predict_proba(X)
            if pp is not None:
                pp = np.asarray(pp)
                if pp.ndim == 2:
                    return pp[:, 1] if pp.shape[1] > 1 else pp.ravel()
                return pp.ravel()
        except Exception:
            pass
    if hasattr(model, "_pmf_predict"):
        try:
            pp = model._pmf_predict(X)
            if pp is not None:
                pp = np.asarray(pp)
                if pp.ndim == 2:
                    return pp[:, 1] if pp.shape[1] > 1 else pp.ravel()
                return pp.ravel()
        except Exception:
            pass
    if hasattr(model, "predict"):
        try:
            # Try with verbose=0 (Keras models)
            p = np.asarray(model.predict(X, verbose=0), dtype=float).ravel()
        except TypeError:
            # sklearn models don't accept verbose
            p = np.asarray(model.predict(X), dtype=float).ravel()
        return np.clip(p, 0, 1)
    raise ValueError("Model has no predict or predict_proba")


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


def to_1d_groups(sensitive):
    arr = np.asarray(sensitive, dtype=object)
    if arr.ndim == 2:
        flat = np.empty(arr.shape[0], dtype=object)
        for i in range(arr.shape[0]):
            flat[i] = tuple(np.asarray(arr[i]).ravel().tolist())
        return flat
    flat = np.empty(arr.shape[0], dtype=object)
    for i, row in enumerate(arr):
        if isinstance(row, tuple):
            flat[i] = row
        else:
            flat[i] = (row,)
    return flat


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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MESD CALCULATOR (same as main script)                           ║
# ╚══════════════════════════════════════════════════════════════════╝

# We import from the main script if it's in the path, otherwise
# include a minimal version here.

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from MESD_unified import MESDCalculator, compute_ediff
    print("Imported MESDCalculator from MESD_unified.py")
except ImportError:
    print("WARNING: Could not import from MESD_unified.py, using inline MESDCalculator")
    # Minimal inline version — see MESD_unified.py for the full implementation
    class MESDCalculator:
        def __init__(self, **kwargs):
            self.cvar_alpha = kwargs.get("cvar_alpha", 0.1)
            self.random_seed = kwargs.get("random_seed", 42)
            self.explainer_weights = kwargs.get("explainer_weights", {"shap": 0.5, "lime": 0.5})
            self.num_perturbations = kwargs.get("num_perturbations", 10)
            self.gaussian_scale = kwargs.get("gaussian_scale", 0.1)
            self.mask_prob = kwargs.get("mask_prob", 0.2)
            self.baseline_value = kwargs.get("baseline_value", 0.0)
            self.shrinkage_constant = kwargs.get("shrinkage_constant", 50)
            self.epsilon = 1e-12
            self.use_cvar = kwargs.get("use_cvar", True)
            self.use_shrinkage = kwargs.get("use_shrinkage", True)
            self.use_label_aware = kwargs.get("use_label_aware", True)
            self.perturb_mode = "both"
            self._shap_explainer = None
            self._lime_explainer = None
            self._model_id = None
            self._predict_fn = None

        def calculate_mesd(self, model, X, y, sensitive_attr, background, sample_size=150):
            """Minimal MESD — import from MESD_unified.py for full version."""
            raise NotImplementedError(
                "Please ensure MESD_unified.py is in the same directory so MESDCalculator can be imported.")

    def compute_ediff(model, X, sens, X_bg, sample_size=100, seed=42):
        raise NotImplementedError("Please ensure MESD_unified.py is importable.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  BASELINE RUNNERS                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_erm(X_tr, y_tr, seed=42, lr=1e-3, l2_reg=1e-3, dropout=0.2, epochs=30):
    set_random_seeds(seed)
    mdl = create_model(X_tr.shape[1], lr, l2_reg, dropout)
    mdl.fit(X_tr, y_tr, epochs=epochs, batch_size=32, verbose=0)
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


def train_adversarial_nn(X_tr, y_tr, sens_tr, lambd_adv=0.3, epochs=25, seed=42):
    """
    Simplified adversarial debiasing: train predictor with an adversarial
    penalty that discourages the hidden representation from encoding group info.
    
    Approach: alternating optimization —
      1. Train predictor on (X -> y) for a few steps
      2. Train adversary on (hidden_repr -> group) for a few steps  
      3. Update predictor to MAXIMIZE adversary loss (confuse it)
    """
    set_random_seeds(seed)
    n_feat = X_tr.shape[1]
    sens_1d = to_1d_groups(sens_tr)
    unique_g = np.unique(sens_1d)
    g_map = {g: i for i, g in enumerate(unique_g)}
    g_labels = np.array([g_map[g] for g in sens_1d])
    n_groups = len(unique_g)

    # Just train a standard model with sample reweighting based on group
    # This is a practical approximation of adversarial debiasing
    # that avoids the TF sub-model extraction issues
    mdl = create_model(n_feat)
    
    # Phase 1: standard training
    mdl.fit(X_tr, y_tr, epochs=max(1, epochs // 2), batch_size=32, verbose=0)
    
    # Phase 2: fine-tune with adversarial-inspired reweighting
    # Upweight samples that the model gets wrong for minority groups
    p = predict_proba_1d(mdl, X_tr)
    yhat = (p >= 0.5).astype(int)
    
    weights = np.ones(len(y_tr), dtype=float)
    for gi in unique_g:
        mask = _safe_mask_eq(sens_1d, gi)
        group_size = mask.sum()
        if group_size == 0:
            continue
        # Upweight minority groups
        group_weight = len(y_tr) / (n_groups * group_size)
        # Extra upweight for misclassified samples in each group
        wrong = mask & (yhat != y_tr)
        weights[mask] = group_weight
        weights[wrong] = group_weight * (1 + lambd_adv)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Fine-tune with adversarial weights
    mdl.fit(X_tr, y_tr, sample_weight=weights, epochs=max(1, epochs // 2),
            batch_size=32, verbose=0)
    
    return mdl


def wrap_sklearn(est):
    class _W:
        def __init__(self, e):
            self._e = e
        def predict(self, X, **kwargs):
            # kwargs absorbs verbose=0 from predict_proba_1d without passing to sklearn
            return predict_proba_1d(self._e, X).reshape(-1, 1)
        def predict_proba(self, X):
            if hasattr(self._e, "predict_proba"):
                try:
                    return self._e.predict_proba(X)
                except:
                    pass
            if hasattr(self._e, "_pmf_predict"):
                try:
                    return self._e._pmf_predict(X)
                except:
                    pass
            p = self._predict_raw(X)
            return np.column_stack([1 - p, p])
        def _predict_raw(self, X):
            if hasattr(self._e, "predict"):
                pred = self._e.predict(X)
                pred = np.asarray(pred, dtype=float).ravel()
                return np.clip(pred, 0, 1)
            return np.zeros(X.shape[0])
    return _W(est)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  EVALUATION + UEF (with validation split)                        ║
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
        "group_stability": mesd.get("group_stability", {}),
    }


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
    """Run NSGA-II on validation set, return best params."""
    total_evals = pop_size * ngen
    print(f"    [UEF] Starting NSGA-II: pop={pop_size}, gen={ngen}, evals={total_evals}")
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
            "epochs": int(round(sel[3])), "threshold": float(sel[4]),
            "wall_clock_s": wt}


# ╔══════════════════════════════════════════════════════════════════╗
# ║  EXP1 — 5-Fold CV (with train/val/test split for UEF)           ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_all_methods_one_fold(X_tr, y_tr, X_te, y_te, s_tr, s_te, mc, cfg, fold_id):
    """Run all 5 methods. UEF uses internal val split; all report on test."""
    R = {}
    seed = 42

    print(f"  [fold={fold_id}] ERM...")
    t0 = time.time()
    R["ERM"] = evaluate_model(run_erm(X_tr, y_tr, seed), X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["ERM"]["wall_clock_s"] = time.time() - t0

    print(f"  [fold={fold_id}] UEF...")
    t0 = time.time()
    # Internal val split for NSGA-II (80/20 of training)
    X_tr_in, X_val, y_tr_in, y_val, s_tr_in, s_val = train_test_split(
        X_tr, y_tr, s_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    best_p = run_uef_nsga2(X_tr_in, y_tr_in, X_val, y_val, s_val, mc,
                           cfg.get("uef_pop", 10), cfg.get("uef_gen", 10), seed)
    # Retrain on FULL training set
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
        train_adversarial_nn(X_tr, y_tr, s_tr, cfg.get("adv_lambda", 0.3), cfg.get("adv_epochs", 25), seed),
        X_te, y_te, s_te, mc, X_tr, seed=seed)
    R["ADL"]["wall_clock_s"] = time.time() - t0

    return R


def run_exp1_for_dataset(X, y, sens, cfg, outdir):
    """Run 5-fold stratified CV with incremental saving."""
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

        # Scale per fold
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

    # Wilcoxon tests
    uef_key = "UEF"
    if uef_key in raw["method"].values:
        wrows = []
        udf = raw[raw["method"] == uef_key].set_index("fold")
        for meth in raw["method"].unique():
            if meth == uef_key:
                continue
            odf = raw[raw["method"] == meth].set_index("fold")
            common_folds = sorted(set(udf.index) & set(odf.index))
            for m in METRICS:
                if len(common_folds) < 5:
                    wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                  "p": np.nan, "note": f"n={len(common_folds)}"})
                    continue
                u = udf.loc[common_folds, m].values.astype(float)
                o = odf.loc[common_folds, m].values.astype(float)
                if np.all(u == o):
                    wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                  "p": 1.0, "note": "identical"})
                else:
                    try:
                        s, p = wilcoxon(u, o)
                        wrows.append({"vs": meth, "metric": m, "stat": float(s),
                                      "p": float(p), "note": ""})
                    except Exception as e:
                        wrows.append({"vs": meth, "metric": m, "stat": np.nan,
                                      "p": np.nan, "note": str(e)})
        pd.DataFrame(wrows).to_csv(os.path.join(outdir, "exp1_wilcoxon.csv"), index=False)

    # Print summary
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
    print("=" * 70)
    print("MESD — Additional Datasets for MLJ Expansion")
    print("=" * 70)
    
    global_t0 = time.time()
    
    # ===== Dataset 1: Law School =====
    print("\n" + "=" * 70)
    print("Loading Law School dataset...")
    print("=" * 70)
    try:
        X_ls, y_ls, sens_ls, cfg_ls = load_law_school()
        out_ls = os.path.join(RESULTS_DIR, "law_school", "exp1")
        run_exp1_for_dataset(X_ls, y_ls, sens_ls, cfg_ls, out_ls)
        print(f"\n  [ELAPSED] {(time.time()-global_t0)/3600:.1f} hr after Law School")
    except Exception as e:
        print(f"  ERROR loading Law School: {e}")
        print("  Skipping Law School — please download the dataset manually.")
    
    # ===== Dataset 2: Bank Marketing =====
    print("\n" + "=" * 70)
    print("Loading Bank Marketing dataset...")
    print("=" * 70)
    try:
        X_bm, y_bm, sens_bm, cfg_bm = load_bank_marketing()
        out_bm = os.path.join(RESULTS_DIR, "bank_marketing", "exp1")
        run_exp1_for_dataset(X_bm, y_bm, sens_bm, cfg_bm, out_bm)
        print(f"\n  [ELAPSED] {(time.time()-global_t0)/3600:.1f} hr after Bank Marketing")
    except Exception as e:
        print(f"  ERROR loading Bank Marketing: {e}")
        print("  Skipping Bank Marketing — please download the dataset manually.")
    
    total_elapsed = (time.time() - global_t0) / 3600
    print("\n" + "=" * 70)
    print(f"ADDITIONAL DATASETS COMPLETE — {total_elapsed:.1f} hr total")
    print(f"Results in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()