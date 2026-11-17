"""
MESD Experiments (All datasets, updated RQ1 & RQ2)

RQ1 (updated):
  Compare UEF (NSGA-II Pareto + Chebyshev) against three fairness baselines:
    - Adversarial learning (in-processing, gradient reversal)
    - Reductions (Agarwal et al.-style reductions; ExponentiatedGradient)
    - Reweighing (Kamiran-Calderes style sample reweighting)

RQ2 (updated):
  For each dataset, produce 4 plots comparing UEF and baselines on the THREE objectives:
    (1) 1 - AUC   (minimize)
    (2) DP_int (intersectional demographic parity max-gap) (minimize)
    (3) MESD_CVaR (minimize)
  Plots:
    - DP_int vs (1-AUC)
    - MESD_CVaR vs (1-AUC)
    - DP_int vs MESD_CVaR
    - 3D scatter: (1-AUC, DP_int, MESD_CVaR)

Also:
  - Keep protected attributes in X (to prevent proxy bias), except where explicitly dropped.
  - Save per-intersectional-group stability CSV for each method run.
"""

import os, random, warnings, functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# ==== ML ====
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

# ==== Reductions baseline ====
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# ==== Explainability ====
import shap
from lime.lime_tabular import LimeTabularExplainer

# ==== Keras / TF ====
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ==== NSGA-II (pymoo) ====
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


def predict_proba_1d(model, X):
    """Return 1D probability/score for positive class for both Keras and sklearn models."""
    # sklearn-style
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].reshape(-1)
        return p.reshape(-1)
    # keras-style (or anything with predict)
    try:
        p = model.predict(X, verbose=0)
    except TypeError:
        p = model.predict(X)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1].reshape(-1)
    return p.reshape(-1)

def to_fairlearn_labels(sensitive_features):
    """Fairlearn reductions expects a 1D array of scalar group labels."""
    g = to_1d_groups(sensitive_features)
    labels = []
    for item in g:
        if isinstance(item, tuple):
            labels.append("|".join(map(str, item)))
        elif isinstance(item, (list, np.ndarray)):
            labels.append("|".join(map(str, np.asarray(item).ravel().tolist())))
        else:
            labels.append(str(item))
    return np.asarray(labels, dtype=str)

RESULTS_DIR = "results_all_RQs"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------
# Reproducibility helpers
# ------------------------
def set_random_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ------------------------
# Group helpers (broadcast-safe)
# ------------------------
def to_1d_groups(sensitive):
    """
    Force sensitive_features into a 1D object array where each entry is a tuple ID.
    Works for:
      - (n,k) numpy arrays / pandas DataFrames
      - 1D arrays of tuples
      - 1D arrays of lists/arrays
    """
    if isinstance(sensitive, pd.DataFrame):
        arr = sensitive.values
    else:
        arr = np.asarray(sensitive, dtype=object)

    if arr.ndim == 2:
        flat = np.empty(arr.shape[0], dtype=object)
        for i in range(arr.shape[0]):
            flat[i] = tuple(np.asarray(arr[i]).ravel().tolist())
        return flat

    # 1D
    flat = np.empty(arr.shape[0], dtype=object)
    for i, row in enumerate(arr):
        if isinstance(row, tuple):
            flat[i] = row
        elif isinstance(row, (list, np.ndarray)):
            flat[i] = tuple(np.asarray(row).ravel().tolist())
        else:
            # scalar (should not happen for your intersectional case, but safe)
            flat[i] = (row,)
    return flat

def _safe_mask_eq_1d_object(sens_1d, g):
    # Avoid numpy broadcasting; do python tuple-equality per element
    return np.array([s == g for s in sens_1d], dtype=bool)

# ------------------------
# Intersectional fairness metrics (max-gap)
# ------------------------
def dp_intersectional_maxgap(y_true, y_pred, sensitive_features):
    sens = to_1d_groups(sensitive_features)
    y_pred = np.asarray(y_pred).reshape(-1)
    rates = []
    for g in np.unique(sens):
        m = _safe_mask_eq_1d_object(sens, g)
        if m.sum() == 0:
            continue
        rates.append(float(np.mean(y_pred[m] == 1)))
    if len(rates) <= 1:
        return 0.0
    return float(np.max(rates) - np.min(rates))

def eo_intersectional_maxgap(y_true, y_pred, sensitive_features):
    sens = to_1d_groups(sensitive_features)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    tprs, fprs = [], []
    for g in np.unique(sens):
        mg = _safe_mask_eq_1d_object(sens, g)
        if mg.sum() == 0:
            continue

        pos = mg & (y_true == 1)
        neg = mg & (y_true == 0)

        if pos.sum() > 0:
            tprs.append(float(np.mean(y_pred[pos] == 1)))
        if neg.sum() > 0:
            fprs.append(float(np.mean(y_pred[neg] == 1)))

    if len(tprs) <= 1:
        tpr_gap = 0.0
    else:
        tpr_gap = float(np.max(tprs) - np.min(tprs))

    if len(fprs) <= 1:
        fpr_gap = 0.0
    else:
        fpr_gap = float(np.max(fprs) - np.min(fprs))

    return float(max(tpr_gap, fpr_gap))

# ------------------------
# Model (paper-1 style NN)
# ------------------------
def create_model(input_dim, learning_rate, l2_reg=0.01, dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,), kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(32, activation="relu", kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=float(learning_rate)), loss="binary_crossentropy")
    return model

# ------------------------
# Gradient reversal (Adversarial baseline)
# ------------------------
class GradientReversal(Layer):
    def __init__(self, lambd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = float(lambd)

    def call(self, x):
        lambd = self.lambd

        @tf.custom_gradient
        def _flip_grad(x):
            def grad(dy):
                return -lambd * dy
            return x, grad

        return _flip_grad(x)

def train_adversarial_nn(X_train, y_train, sens_train, *,
                         lr=1e-3, l2_reg=1e-3, dropout=0.2,
                         epochs=20, batch_size=64, lambd_adv=0.5, seed=42):
    """
    Simple adversarial debiasing:
      - predictor learns Y
      - adversary tries to predict intersectional group G from representation
      - GRL flips gradient to remove group info
    """
    set_random_seeds(seed)

    sens = to_1d_groups(sens_train)
    # map group tuples to integer class
    uniq = {g:i for i,g in enumerate(np.unique(sens))}
    g_idx = np.array([uniq[s] for s in sens], dtype=int)
    n_groups = len(uniq)

    inp = Input(shape=(X_train.shape[1],), name="x")
    h = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(inp)
    h = Dropout(dropout)(h)
    h = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(h)

    # predictor head
    y_out = Dense(1, activation="sigmoid", name="y_hat")(h)

    # adversary head (with gradient reversal)
    grl = GradientReversal(lambd=lambd_adv)(h)
    a = Dense(32, activation="relu")(grl)
    g_out = Dense(n_groups, activation="softmax", name="g_hat")(a)

    model = Model(inputs=inp, outputs=[y_out, g_out])
    model.compile(
        optimizer=Adam(learning_rate=float(lr)),
        loss={"y_hat": "binary_crossentropy", "g_hat": "sparse_categorical_crossentropy"},
        loss_weights={"y_hat": 1.0, "g_hat": 1.0},  # GRL already flips sign; keep weights comparable
    )

    model.fit(
        X_train,
        {"y_hat": y_train, "g_hat": g_idx},
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        shuffle=True
    )

    # return a "predictor-only" model with same input->y_hat
    predictor = Model(inputs=inp, outputs=y_out)
    predictor.compile(optimizer=Adam(learning_rate=float(lr)), loss="binary_crossentropy")
    return predictor, uniq

# ------------------------
# Reweighing baseline
# ------------------------
def compute_reweighing_weights(y, sens):
    """
    Kamiran & Calders style reweighing on (A,Y) cells.
    sens should be 1D tuple group IDs.
    """
    y = np.asarray(y).reshape(-1)
    sens = to_1d_groups(sens)

    n = len(y)
    # counts
    unique_a = np.unique(sens)
    unique_y = np.unique(y)

    # P(A), P(Y), P(A,Y)
    p_a = {a: 0 for a in unique_a}
    p_y = {yy: 0 for yy in unique_y}
    p_ay = {(a,yy): 0 for a in unique_a for yy in unique_y}

    for i in range(n):
        a = sens[i]
        yy = int(y[i])
        p_a[a] += 1
        p_y[yy] += 1
        p_ay[(a,yy)] += 1

    for a in p_a: p_a[a] /= n
    for yy in p_y: p_y[yy] /= n
    for k in p_ay: p_ay[k] /= n

    w = np.zeros(n, dtype=float)
    for i in range(n):
        a = sens[i]
        yy = int(y[i])
        denom = p_ay[(a,yy)]
        if denom <= 0:
            w[i] = 1.0
        else:
            w[i] = (p_a[a] * p_y[yy]) / denom
    # normalize to mean 1 (helps NN training)
    w = w / (np.mean(w) + 1e-12)
    return w

# ------------------------
# MESD Calculator (uses inverted L2 similarity)
# ------------------------
class MESDCalculator:
    def __init__(self,
                 explainer_weights=None,
                 num_perturbations=10,
                 perturb_mode="both",
                 gaussian_scale=0.1,
                 mask_prob=0.2,
                 baseline_value=0.0,
                 num_bins=5,
                 cvar_alpha=0.1,
                 min_cell_size=10,
                 shrinkage_constant=50,
                 epsilon=1e-12,
                 random_seed=42):
        self.explainer_weights = explainer_weights or {"shap": 0.5, "lime": 0.5}
        wsum = sum(self.explainer_weights.values())
        self.explainer_weights = {k: v/(wsum + epsilon) for k, v in self.explainer_weights.items()}

        self.num_perturbations = int(num_perturbations)
        self.perturb_mode = perturb_mode
        self.gaussian_scale = float(gaussian_scale)
        self.mask_prob = float(mask_prob)
        self.baseline_value = float(baseline_value)
        self.num_bins = None if num_bins in (None, 0) else int(num_bins)

        self.cvar_alpha = float(cvar_alpha)
        self.min_cell_size = int(min_cell_size)
        self.shrinkage_constant = float(shrinkage_constant)
        self.epsilon = float(epsilon)
        self.random_seed = int(random_seed)

        self._shap_explainer = None
        self._lime_explainer = None
        self._model_id = None
        self._predict_fn = None

    def norm_l1(self, v):
        denom = np.sum(np.abs(v)) + self.epsilon
        return v / denom

    def dist_to_similarity(self, d):
        return 1.0 / (1.0 + float(d))

    def ensemble_attribution(self, attrs):
        base = np.zeros_like(list(attrs.values())[0])
        for k, v in attrs.items():
            if k in self.explainer_weights:
                base += self.explainer_weights[k] * v
        return base

    def perturb_instance(self, x):
        z = x.copy()
        if self.perturb_mode in ("gaussian", "both"):
            z = z + np.random.normal(0, self.gaussian_scale, size=z.shape[0])
        if self.perturb_mode in ("masking", "both"):
            mask = np.random.rand(z.shape[0]) < self.mask_prob
            z[mask] = self.baseline_value
        return z

    def _initialize_explainers(self, model, background, n_features):
        bg = np.asarray(background)
        try:
            self._shap_explainer = shap.DeepExplainer(model, bg)
        except Exception as e1:
            print(f"  [MESD] SHAP DeepExplainer failed: {e1}")
            try:
                self._shap_explainer = shap.GradientExplainer(model, bg)
            except Exception as e2:
                print(f"  [MESD] SHAP GradientExplainer failed: {e2}")
                self._shap_explainer = None

        # Fallback for linear sklearn models (e.g., LogisticRegression)
        if self._shap_explainer is None and hasattr(model, "coef_"):
            try:
                self._shap_explainer = shap.LinearExplainer(model, bg)
            except Exception as e3:
                print(f"  [MESD] SHAP LinearExplainer failed: {e3}")
                self._shap_explainer = None

        try:
            self._lime_explainer = LimeTabularExplainer(
                training_data=bg,
                feature_names=[f"F{i}" for i in range(n_features)],
                class_names=["class0", "class1"],
                mode="classification"
            )

            def predict_fn(X):
                p = predict_proba_1d(model, X)
                out = np.zeros((X.shape[0], 2))
                out[:, 0] = 1 - p
                out[:, 1] = p
                return out

            self._predict_fn = predict_fn
        except Exception as e:
            print(f"  [MESD] LIME init failed: {e}")
            self._lime_explainer, self._predict_fn = None, None

    def get_shap(self, x):
        if self._shap_explainer is None:
            return np.zeros(x.shape[0])
        try:
            sv = self._shap_explainer.shap_values(x.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[0]
            v = np.squeeze(sv).reshape(-1)
            return self.norm_l1(v)
        except Exception:
            return np.zeros(x.shape[0])

    def get_lime(self, x, n_features):
        if self._lime_explainer is None or self._predict_fn is None:
            return np.zeros(n_features)
        try:
            exp = self._lime_explainer.explain_instance(
                data_row=x,
                predict_fn=self._predict_fn,
                num_features=n_features,
                top_labels=1
            )
            weights = dict(exp.as_list(label=1))
            arr = np.zeros(n_features)
            # LIME feature names are "F{i}"
            for name, w in weights.items():
                if name.startswith("F"):
                    j = int(name[1:])
                    if 0 <= j < n_features:
                        arr[j] = float(w)
            return self.norm_l1(arr)
        except Exception:
            return np.zeros(n_features)

    def stability_score(self, attributions_list):
        # similarity in (0,1], higher better
        if len(attributions_list) <= 1:
            return 1.0
        base = attributions_list[0]
        total = 0.0
        cnt = 0
        for k in range(1, len(attributions_list)):
            d = float(np.linalg.norm(base - attributions_list[k]))
            total += self.dist_to_similarity(d)
            cnt += 1
        return float(total / max(cnt, 1))

    def calculate_mesd(self, model, X, y, sensitive_attr, background, sample_size=150):
        # Reset explainers when model changes so each method uses its own SHAP/LIME
        if self._model_id is None or self._model_id != id(model):
            self._shap_explainer = None
            self._lime_explainer = None
            self._model_id = id(model)

        np.random.seed(self.random_seed)

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        sens = to_1d_groups(sensitive_attr)

        n = X.shape[0]
        m = X.shape[1]
        if n == 0:
            return {"mesd_cvar": 0.0, "mesd_max": 0.0, "mesd_var": 0.0, "group_stability": {}}

        idx = np.random.choice(n, size=min(sample_size, n), replace=False)
        Xs = X[idx]
        ys = y[idx]
        gs = sens[idx]

        if self._lime_explainer is None and self._shap_explainer is None:
            self._initialize_explainers(model, np.asarray(background), m)

        # per instance stability
        stability = []
        attrs_for_instance = []
        for i in range(Xs.shape[0]):
            x0 = Xs[i]
            attrs_list = []
            # original
            attrs = {
                "shap": self.get_shap(x0),
                "lime": self.get_lime(x0, m),
            }
            a0 = self.ensemble_attribution(attrs)
            attrs_list.append(a0)

            # perturbations
            for _ in range(self.num_perturbations):
                xp = self.perturb_instance(x0)
                attrs_p = {
                    "shap": self.get_shap(xp),
                    "lime": self.get_lime(xp, m),
                }
                ap = self.ensemble_attribution(attrs_p)
                attrs_list.append(ap)

            s = self.stability_score(attrs_list)
            stability.append(s)
            attrs_for_instance.append(attrs_list[0])

        stability = np.asarray(stability, float)

        # Aggregate by group and label (broadcast-safe)
        unique_groups = np.unique(gs)
        unique_labels = np.unique(ys)

        gl_mean = {}
        gl_count = {}
        Sg = {}

        for g in unique_groups:
            mg = _safe_mask_eq_1d_object(gs, g)
            if mg.sum() == 0:
                continue
            Sg[g] = float(np.mean(stability[mg]))

        for g in unique_groups:
            mg = _safe_mask_eq_1d_object(gs, g)
            for yy in unique_labels:
                mgy = mg & (ys == yy)
                c = int(np.sum(mgy))
                if c == 0:
                    continue
                gl_count[(g, int(yy))] = c
                gl_mean[(g, int(yy))] = float(np.mean(stability[mgy]))

        # Compute pairwise MESD disparities over all group pairs
        groups_list = list(Sg.keys())
        if len(groups_list) <= 1:
            return {
                "group_stability": Sg,
                "group_label_stability": gl_mean,
                "group_label_counts": gl_count,
                "mesd_max": 0.0,
                "mesd_var": 0.0,
                "mesd_cvar": 0.0,
            }

        pair_disp = []
        pair_risk = []
        pair_ids = []
        for i in range(len(groups_list)):
            for j in range(i+1, len(groups_list)):
                g1, g2 = groups_list[i], groups_list[j]
                d = abs(Sg[g1] - Sg[g2])
                # worst-group risk proxy
                r = 1.0 - min(Sg[g1], Sg[g2])
                pair_disp.append(d)
                pair_risk.append(r)
                pair_ids.append((g1, g2))

        pair_disp = np.asarray(pair_disp, float)
        pair_risk = np.asarray(pair_risk, float)

        mesd_max = float(np.max(pair_disp))
        mesd_var = float(np.mean(pair_disp**2) - np.mean(pair_disp)**2) if len(pair_disp) > 1 else 0.0

        # CVaR weighting (tail on risk)
        alpha = self.cvar_alpha
        if alpha <= 0 or alpha >= 1:
            # uniform weights
            w = np.ones_like(pair_disp) / max(len(pair_disp), 1)
            mesd_cvar = float(np.sum(w * pair_disp))
            weights = w
        else:
            tau = np.quantile(pair_risk, 1.0 - alpha)
            tail = (pair_risk >= tau).astype(float)
            w = tail / (np.sum(tail) + self.epsilon)
            mesd_cvar = float(np.sum(w * pair_disp))
            weights = w

        return {
            "group_stability": Sg,
            "group_label_stability": gl_mean,
            "group_label_counts": gl_count,
            "mesd_max": mesd_max,
            "mesd_var": mesd_var,
            "mesd_cvar": mesd_cvar,
            "pair_ids": pair_ids,
            "pair_weights": weights,
        }

# ------------------------
# Common evaluation wrapper
# ------------------------
def evaluate_model(model, X_test, y_test, sens_test, mesd_calc, X_train_bg, threshold=0.5, seed=42):
    set_random_seeds(seed)
    # probability
    p = predict_proba_1d(model, X_test)
    yhat = (p >= float(threshold)).astype(int)

    auc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else 0.5
    f1 = f1_score(y_test, yhat) if len(np.unique(y_test)) > 1 else 0.0
    dp = dp_intersectional_maxgap(y_test, yhat, sens_test)
    eo = eo_intersectional_maxgap(y_test, yhat, sens_test)

    mesd = mesd_calc.calculate_mesd(model, X_test, y_test, sens_test, X_train_bg, sample_size=150)
    return {
        "auc": float(auc),
        "f1": float(f1),
        "dp": float(dp),
        "eo": float(eo),
        "mesd_cvar": float(mesd["mesd_cvar"]),
        "mesd_max": float(mesd["mesd_max"]),
        "mesd_var": float(mesd["mesd_var"]),
        "group_stability": mesd["group_stability"],
        "mesd_detail": mesd,
        "proba": p,
        "yhat": yhat
    }

# ------------------------
# UEF via NSGA-II + Chebyshev selection
# ------------------------
def chebyshev_select(F, ideal, nadir, weights=None):
    F = np.asarray(F, float)
    ideal = np.asarray(ideal, float)
    nadir = np.asarray(nadir, float)
    denom = np.maximum(nadir - ideal, 1e-12)
    Z = (F - ideal) / denom
    w = np.ones(F.shape[1]) / F.shape[1] if weights is None else np.asarray(weights, float)
    scores = np.max(w * Z, axis=1)
    return int(np.argmin(scores))

class NNHyperparamProblem(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, sens_test, mesd_calc, seed=42):
        xl = np.array([1e-5, 1e-4, 0.0, 10, 0.2])
        xu = np.array([5e-3, 5e-2, 0.5, 60, 0.8])
        super().__init__(n_var=5, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.sens_test = sens_test
        self.mesd_calc = mesd_calc
        self.seed = seed

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        F = []
        for row in X:
            lr, l2r, dr, ep, th = row
            model = create_model(self.X_train.shape[1], float(lr), float(l2r), float(dr))
            model.fit(self.X_train, self.y_train, epochs=int(round(ep)), batch_size=32, verbose=0)
            metrics = evaluate_model(model, self.X_test, self.y_test, self.sens_test,
                                     self.mesd_calc, self.X_train, threshold=float(th), seed=self.seed)
            # objectives to MINIMIZE:
            F.append([1.0 - metrics["auc"], metrics["dp"], metrics["mesd_cvar"]])
        out["F"] = np.asarray(F, float)

def run_uef_nsga2(X_train, y_train, X_test, y_test, sens_test, mesd_calc,
                  pop_size=15, ngen=15, seed=42):
    problem = NNHyperparamProblem(X_train, y_train, X_test, y_test, sens_test, mesd_calc, seed=seed)
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", ngen)

    res = minimize(problem, algorithm, termination, seed=seed, save_history=False, verbose=False)

    F = res.F
    X = res.X

    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    k = chebyshev_select(F, ideal, nadir, weights=np.array([1/3, 1/3, 1/3]))

    sel = X[k]
    params = {"lr": float(sel[0]), "l2": float(sel[1]), "dropout": float(sel[2]),
              "epochs": int(round(sel[3])), "threshold": float(sel[4])}

    # retrain selected
    model = create_model(X_train.shape[1], params["lr"], params["l2"], params["dropout"])
    model.fit(X_train, y_train, epochs=int(params["epochs"]), batch_size=32, verbose=0)
    chosen = evaluate_model(model, X_test, y_test, sens_test, mesd_calc, X_train, threshold=params["threshold"], seed=seed)

    return {"params": params, "chosen": chosen, "pareto_F": F, "pareto_X": X}

# ------------------------
# Baselines
# ------------------------
def run_reweighing(X_train, y_train, sens_train, X_test, y_test, sens_test, mesd_calc, seed=42,
                   lr=1e-3, l2_reg=1e-3, dropout=0.2, epochs=30):
    set_random_seeds(seed)
    w = compute_reweighing_weights(y_train, sens_train)
    model = create_model(X_train.shape[1], lr, l2_reg, dropout)
    model.fit(X_train, y_train, sample_weight=w, epochs=int(epochs), batch_size=32, verbose=0)
    return model

def run_reductions(X_train, y_train, sens_train, eps=0.02, seed=42):
    # LogisticRegression base estimator
    set_random_seeds(seed)
    sens_1d = to_fairlearn_labels(sens_train)
    base = LogisticRegression(max_iter=1000, solver="lbfgs")
    eg = ExponentiatedGradient(base, DemographicParity(), eps=float(eps))
    eg.fit(X_train, y_train, sensitive_features=sens_1d)
    return eg  # has predict()

def predict_proba_from_reductions(est, X):
    # ExponentiatedGradient returns a randomized classifier with predict(). No reliable proba.
    # We approximate with predicted label as probability (0/1).
    yhat = est.predict(X)
    return np.asarray(yhat, float)

def wrap_reductions_as_keras_predictor(est):
    # Adapter so evaluate_model can call .predict(X, verbose=0)
    class _Wrap:
        def predict(self, X, verbose=0):
            p = predict_proba_from_reductions(est, X)
            return p.reshape(-1, 1)
    return _Wrap()

def run_adversarial(X_train, y_train, sens_train, X_test, y_test, sens_test, mesd_calc, seed=42,
                    lambd_adv=0.5, epochs=25):
    predictor, _ = train_adversarial_nn(X_train, y_train, sens_train,
                                        lr=1e-3, l2_reg=1e-3, dropout=0.2,
                                        epochs=epochs, batch_size=64,
                                        lambd_adv=float(lambd_adv), seed=seed)
    return predictor

# ------------------------
# Data loading / encoding
# ------------------------
def encode_binary_column(series, mapping):
    """Encode a binary column robustly.

    Handles common issues:
      - case/whitespace differences in string categories (e.g., 'Adult' vs 'adult')
      - numeric categories stored as strings (e.g., '1', '2')
      - raises a clear error if any values are unmapped
    """
    s = series.copy()

    # Build a normalized mapping for string-like values
    norm_map = {}
    for k, v in mapping.items():
        if isinstance(k, str):
            norm_map[k.strip().lower()] = v
        else:
            norm_map[str(k).strip().lower()] = v

    # Normalize series to string for mapping when object-like
    if s.dtype == object or isinstance(s.iloc[0], str):
        s_norm = s.astype(str).str.strip().str.lower()
        out = s_norm.map(norm_map)

        # Fallback: try direct mapping on original series (covers numeric keys)
        if out.isna().any():
            out2 = s.map(mapping)
            out = out.where(~out.isna(), out2)
    else:
        out = s.map(mapping)

    # If anything is still unmapped, fail loudly with a helpful message
    if out.isna().any():
        unmapped = pd.unique(s[out.isna()])
        raise ValueError(
            f"encode_binary_column: unmapped values in '{getattr(series, 'name', 'column')}'. "
            f"Examples: {unmapped[:10].tolist()} (showing up to 10). Mapping keys: {list(mapping.keys())}"
        )

    return out.astype(int)


def _resolve_col(df, name):
    """Resolve a column name case-insensitively.
    Raises KeyError with a helpful message if not found.
    """
    cmap = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key in cmap:
        return cmap[key]
    raise KeyError(f"Column '{name}' not found. Available columns: {list(df.columns)}")


def preprocess_dataset(df, *, target_col, target_mapping, protected_cols, protected_mappings,
                       drop_cols=None, seed=42):
    df = df.copy()

    # Resolve columns case-insensitively (German data uses 'sex', not 'Sex', etc.)
    target_actual = _resolve_col(df, target_col)
    protected_actuals = [_resolve_col(df, c) for c in protected_cols]

    # map target (supports string/numeric with normalization inside encode_binary_column)
    df[target_actual] = encode_binary_column(df[target_actual], target_mapping)

    # map protected (use mappings keyed by the user-provided names)
    for req_name, actual_name in zip(protected_cols, protected_actuals):
        if req_name not in protected_mappings:
            raise KeyError(f"Missing protected_mappings for '{req_name}'. Keys: {list(protected_mappings.keys())}")
        df[actual_name] = encode_binary_column(df[actual_name], protected_mappings[req_name])

    drop_cols = drop_cols or []
    # Build X, keeping protected in X unless user told to drop something else.
    X_df = df.drop([target_actual] + drop_cols, axis=1, errors="ignore")
    y = df[target_actual].values.astype(int)

    # One-hot encode object columns
    obj_cols = [c for c in X_df.columns if X_df[c].dtype == object]
    if obj_cols:
        X_df = pd.get_dummies(X_df, columns=obj_cols, drop_first=True)

    # standardize numeric columns
    X = X_df.values.astype(float)

    sens = df[protected_actuals].values  # (n,k) for intersectional grouping
    return X, y, sens, X_df.columns.tolist()


def save_tradeoff_plots(points_df, outdir, dataset_name):
    """
    points_df columns: method, auc, dp, mesd_cvar
    """
    os.makedirs(outdir, exist_ok=True)

    points_df = points_df.copy()
    points_df["one_minus_auc"] = 1.0 - points_df["auc"]

    # 2D plots
    def _scatter(x, y, xlabel, ylabel, fname):
        plt.figure()
        for method, sub in points_df.groupby("method"):
            plt.scatter(sub[x], sub[y], label=method, alpha=0.8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{dataset_name}: {ylabel} vs {xlabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

    _scatter("dp", "one_minus_auc", "DP_int", "1-AUC", "plot_dp_vs_1mauc.png")
    _scatter("mesd_cvar", "one_minus_auc", "MESD_CVaR", "1-AUC", "plot_mesd_vs_1mauc.png")
    _scatter("dp", "mesd_cvar", "DP_int", "MESD_CVaR", "plot_dp_vs_mesd.png")

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for method, sub in points_df.groupby("method"):
        ax.scatter(sub["one_minus_auc"], sub["dp"], sub["mesd_cvar"], label=method, alpha=0.8)
    ax.set_xlabel("1-AUC")
    ax.set_ylabel("DP_int")
    ax.set_zlabel("MESD_CVaR")
    ax.set_title(f"{dataset_name}: 3D objectives")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_3d_objectives.png"), dpi=220)
    plt.close()

def save_group_stability_csv(group_stability_dict, outpath):
    # group keys are tuples; convert to strings
    rows = [{"group": str(g), "stability": float(s)} for g, s in group_stability_dict.items()]
    pd.DataFrame(rows).sort_values("stability", ascending=True).to_csv(outpath, index=False)

# ------------------------
# Main experiment runner
# ------------------------
def run_dataset(dataset_cfg, seed=42):
    name = dataset_cfg["name"]
    path = dataset_cfg["path"]

    print("\n" + "="*70)
    print(f"DATASET: {name}")
    print("="*70)

    df = pd.read_csv(path)

    X, y, sens, feat_names = preprocess_dataset(
        df,
        target_col=dataset_cfg["target_col"],
        target_mapping=dataset_cfg["target_mapping"],
        protected_cols=dataset_cfg["protected_cols"],
        protected_mappings=dataset_cfg["protected_mappings"],
        drop_cols=dataset_cfg.get("drop_cols", []),
        seed=seed
    )

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sens, test_size=0.3, random_state=seed, stratify=y
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    outdir = os.path.join(RESULTS_DIR, name)
    os.makedirs(outdir, exist_ok=True)

    mesd_calc = MESDCalculator(cvar_alpha=0.1, random_seed=seed)

    points = []  # for plots
    summary_rows = []  # RQ1 main table (single chosen model per method)
    rq3_rows = []  # RQ3 table (MESD variants + worst-group stability)

    def _worst_group_info(group_stability: dict):
        """Return (worst_group_key_str, worst_group_stability)."""
        if not group_stability:
            return "", np.nan
        # group_stability keys are tuples; stringify for CSV
        worst_g = min(group_stability, key=lambda k: group_stability[k])
        return str(worst_g), float(group_stability[worst_g])

    def _add_rows(method_name: str, metrics: dict):
        """Append rows for RQ1 and RQ3 using evaluate_model() output."""
        summary_rows.append({
            "dataset": name,
            "method": method_name,
            "auc": float(metrics["auc"]),
            "f1": float(metrics["f1"]),
            "dp": float(metrics["dp"]),
            "eo": float(metrics["eo"]),
            "mesd_cvar": float(metrics["mesd_cvar"]),
            "mesd_max": float(metrics["mesd_max"]),
            "mesd_var": float(metrics["mesd_var"]),
        })

        wg, wg_s = _worst_group_info(metrics.get("group_stability", {}))
        rq3_rows.append({
            "dataset": name,
            "method": method_name,
            "mesd_max": float(metrics["mesd_max"]),
            "mesd_var": float(metrics["mesd_var"]),
            "mesd_cvar": float(metrics["mesd_cvar"]),
            "worst_group": wg,
            "worst_group_stability": wg_s,
            "num_groups": int(len(metrics.get("group_stability", {}))),
        })

    # ---- UEF ----
    print("[UEF] Running NSGA-II + Chebyshev...")
    uef = run_uef_nsga2(X_train, y_train, X_test, y_test, sens_test, mesd_calc,
                        pop_size=dataset_cfg.get("uef_pop", 30),
                        ngen=dataset_cfg.get("uef_gen", 40),
                        seed=seed)

    # add full pareto to points
    pareto = pd.DataFrame(uef["pareto_F"], columns=["one_minus_auc", "dp", "mesd_cvar"])
    pareto["auc"] = 1.0 - pareto["one_minus_auc"]
    pareto["method"] = "UEF"
    points.append(pareto[["method", "auc", "dp", "mesd_cvar"]])

    chosen = uef["chosen"]
    _add_rows("UEF_selected", chosen)
    save_group_stability_csv(chosen["group_stability"], os.path.join(outdir, "group_stability_UEF_selected.csv"))

    # ---- Reweighing baseline ----
    print("[Baseline] Reweighing...")
    rw_model = run_reweighing(X_train, y_train, sens_train, X_test, y_test, sens_test, mesd_calc, seed=seed)
    # threshold sweep for points
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = evaluate_model(rw_model, X_test, y_test, sens_test, mesd_calc, X_train, threshold=th, seed=seed)
        points.append(pd.DataFrame([{"method": f"Reweighing(th={th})", "auc": m["auc"], "dp": m["dp"], "mesd_cvar": m["mesd_cvar"]}]))
    m05 = evaluate_model(rw_model, X_test, y_test, sens_test, mesd_calc, X_train, threshold=0.5, seed=seed)
    _add_rows("Reweighing", m05)
    save_group_stability_csv(m05["group_stability"], os.path.join(outdir, "group_stability_Reweighing.csv"))

    # ---- Reductions baseline (Agarwal) ----
    print("[Baseline] Reductions (ExponentiatedGradient)...")
    for eps in dataset_cfg.get("reductions_eps_grid", [0.01, 0.02, 0.05, 0.1]):
        red = run_reductions(X_train, y_train, sens_train, eps=eps, seed=seed)
        red_wrap = wrap_reductions_as_keras_predictor(red)
        m = evaluate_model(red_wrap, X_test, y_test, sens_test, mesd_calc, X_train, threshold=0.5, seed=seed)
        points.append(pd.DataFrame([{"method": f"Reductions(eps={eps})", "auc": m["auc"], "dp": m["dp"], "mesd_cvar": m["mesd_cvar"]}]))

    # pick a default eps
    eps0 = dataset_cfg.get("reductions_eps", 0.02)
    red = run_reductions(X_train, y_train, sens_train, eps=eps0, seed=seed)
    red_wrap = wrap_reductions_as_keras_predictor(red)
    m = evaluate_model(red_wrap, X_test, y_test, sens_test, mesd_calc, X_train, threshold=0.5, seed=seed)
    _add_rows(f"Reductions(eps={eps0})", m)
    save_group_stability_csv(m["group_stability"], os.path.join(outdir, "group_stability_Reductions.csv"))

    # ---- Adversarial baseline ----
    print("[Baseline] Adversarial learning (GRL)...")
    for lambd in dataset_cfg.get("adv_lambda_grid", [0.0, 0.1, 0.3, 0.7, 1.0]):
        adv_model = run_adversarial(X_train, y_train, sens_train, X_test, y_test, sens_test, mesd_calc,
                                    seed=seed, lambd_adv=lambd, epochs=dataset_cfg.get("adv_epochs", 25))
        m = evaluate_model(adv_model, X_test, y_test, sens_test, mesd_calc, X_train, threshold=0.5, seed=seed)
        points.append(pd.DataFrame([{"method": f"Adversarial(l={lambd})", "auc": m["auc"], "dp": m["dp"], "mesd_cvar": m["mesd_cvar"]}]))
        if abs(lambd - dataset_cfg.get("adv_lambda", 0.3)) < 1e-9:
            _add_rows(f"Adversarial(l={lambd})", m)
            save_group_stability_csv(m["group_stability"], os.path.join(outdir, "group_stability_Adversarial.csv"))

    # ---- Save points & plots ----
    points_df = pd.concat(points, ignore_index=True)
    points_df.to_csv(os.path.join(outdir, "rq2_points_all_methods.csv"), index=False)
    save_tradeoff_plots(points_df, outdir, name)

    # ---- Save summary ----
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "rq1_summary_table.csv"), index=False)

    # ---- Save RQ3 ----
    rq3 = pd.DataFrame(rq3_rows)
    rq3.to_csv(os.path.join(outdir, "rq3_mesd_variants_and_worstgroup.csv"), index=False)

    return {"rq1": summary, "rq3": rq3}



def main():
    seed = 42
    dataset_cfg = {
        "name": "german",
        "path": "/home/n88r684/MESD/germandata1.csv",
        "target_col": "credit",
        "target_mapping": {1: 0, 2: 1},
        "protected_cols": ["Age_Metric", "Sex"],
        "protected_mappings": {
            "Age_Metric": {"young": 0, "adult": 1, 0: 0, 1: 1},
            "Sex": {"male": 0, "female": 1, 0: 0, 1: 1}
        },
        "drop_cols": [],
        "uef_pop": 15, "uef_gen": 15,
        "reductions_eps": 0.02,
        "adv_lambda": 0.3,
    }
    run_dataset(dataset_cfg, seed=seed)
    print("Done.")

if __name__ == "__main__":
    main()
