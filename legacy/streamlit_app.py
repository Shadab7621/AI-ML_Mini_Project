import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, StratifiedKFold
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix, average_precision_score,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Classification Visualizer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL PLOT STYLE
# ═══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.family": "DejaVu Sans",
})

PURPLE   = "#7C3AED"
VIOLET   = "#8B5CF6"
CYAN     = "#8B5CF6"
GREEN    = "#10B981"
AMBER    = "#F59E0B"
RED      = "#EF4444"
SLATE    = "#64748B"

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]          { font-family: 'Inter', sans-serif; }
.main .block-container              { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }

/* ── Metric Cards ─────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1rem 1.2rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,.10); }
.metric-label  { font-size:.72rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.3rem; }
.metric-value  { font-size:1.9rem; font-weight:800; color:#18181b; }
.metric-value.good { color: #059669; }
.metric-value.warn { color: #d97706; }
.metric-value.bad  { color: #dc2626; }
.metric-sub    { font-size:.7rem; color:#94a3b8; margin-top:.15rem; }

/* ── Section Headers ──────────────────────────────────────────────────── */
.section-hdr {
    font-size:1rem; font-weight:700; color:#18181b;
    border-left:4px solid #7c3aed; padding-left:.65rem;
    margin: 1.4rem 0 .8rem 0;
}

/* ── Badges ───────────────────────────────────────────────────────────── */
.badge {
    display:inline-block;
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    color:white; padding:.3rem .9rem; border-radius:20px;
    font-size:.78rem; font-weight:700; margin-bottom:.75rem;
}

/* ── Info / Warning boxes ─────────────────────────────────────────────── */
.info-box  { background:#faf5ff; border:1px solid #e9d5ff; border-radius:8px; padding:.75rem 1rem; color:#6b21a8; font-size:.85rem; margin:.5rem 0; }
.warn-box  { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:.75rem 1rem; color:#92400e; font-size:.85rem; margin:.5rem 0; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0;
}

/* Base text color inside sidebar - sharp dark slate */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] strong {
    color: #0f172a !important;
}

/* Labels on every widget type - bold, high-contrast, crystal clear */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stTextInput label {
    color: #0f172a !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 0.25rem !important;
}

/* Radio & Checkbox options */
[data-testid="stSidebar"] .stRadio label > div,
[data-testid="stSidebar"] .stCheckbox label > div {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Select box - crisp white card style with dark legible text */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: #7c3aed !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
    color: #0f172a !important;
    background: transparent !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stSelectbox svg {
    fill: #475569 !important;
}

/* MultiSelect tag chips */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #ede9fe !important;
    color: #6d28d9 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* Number input / Text input */
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* Sliders */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #7c3aed !important;
    border-color: #7c3aed !important;
    box-shadow: 0 2px 6px rgba(124,58,237,0.4) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: #7c3aed !important;
}

/* Divider lines in sidebar */
[data-testid="stSidebar"] hr {
    border-color: #e2e8f0 !important;
    margin: 0.8rem 0 !important;
}

/* Tooltip / help icons */
[data-testid="stSidebar"] svg[data-testid="stTooltipIcon"] {
    fill: #64748b !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"]      { border-radius:8px 8px 0 0; padding:.45rem 1.1rem; font-weight:500; font-size:.85rem; }
.stTabs [aria-selected="true"]    { background:#7c3aed; color:white !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color:white; border:none; border-radius:8px;
    padding:.55rem 1.5rem; font-weight:700; width:100%;
    transition: all .2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #6d28d9, #5b21b6); transform:translateY(-1px); box-shadow:0 4px 12px rgba(124,58,237,.4); }

/* ── Download button ─────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: white !important; border:none; border-radius:8px; font-weight:600; width:auto;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
NEEDS_SCALING  = {"Support Vector Machine (SVM)", "Logistic Regression", "K-Nearest Neighbors (KNN)"}
TREE_BASED     = {"Random Forest", "Decision Tree", "Gradient Boosting"}
CLASS_NAMES    = ["edible", "poisonous"]
PALETTE        = [PURPLE, CYAN, GREEN, AMBER, RED, "#EC4899", "#F97316"]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _color_class(v, lo=0.75, hi=0.88):
    return "good" if v >= hi else ("warn" if v >= lo else "bad")

def metric_card(label, value, is_pct=True, sub=""):
    fmt  = f"{value:.1%}" if is_pct else f"{value:.4f}"
    cls  = _color_class(value)
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{fmt}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f"<div class='section-hdr'>{title}</div>", unsafe_allow_html=True)

def badge(text):
    st.markdown(f"<span class='badge'>🧠 {text}</span>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    raw  = pd.read_csv("Mushrooms.csv")
    enc  = raw.copy()
    le   = LabelEncoder()
    for col in enc.columns:
        enc[col] = le.fit_transform(enc[col])
    return enc, raw

@st.cache_data
def split_data(df_enc):
    y = df_enc["type"]
    X = df_enc.drop(columns=["type"])
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline(clf_name, clf):
    steps = []
    if clf_name in NEEDS_SCALING:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    m = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC":       (matthews_corrcoef(y_test, y_pred) + 1) / 2,   # normalise to 0-1
        "Cohen κ":   (cohen_kappa_score(y_test, y_pred) + 1) / 2,
    }
    for k in ("ROC-AUC", "Avg Precision"):
        m[k] = None
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            proba = pipe.decision_function(X_test)
        m["ROC-AUC"]      = roc_auc_score(y_test, proba)
        m["Avg Precision"] = average_precision_score(y_test, proba)
    except Exception:
        pass
    return m, y_pred


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_confusion(pipe, X_test, y_test):
    cm   = confusion_matrix(y_test, pipe.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.6, ax=ax, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    # Add per-class accuracy annotation
    for i in range(len(cm)):
        total = cm[i].sum()
        pct   = cm[i, i] / total if total else 0
        ax.text(len(cm) + 0.05, i + 0.5, f"{pct:.1%}", va="center",
                fontsize=9, color=SLATE)
    plt.tight_layout()
    return fig


def fig_roc(pipe, X_test, y_test):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    display = RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax,
                                              color=PURPLE, lw=2.2, alpha=.9)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.4, label="Random (AUC=0.50)")
    try:
        xd = display.line_.get_xdata()
        yd = display.line_.get_ydata()
        ax.fill_between(xd, yd, alpha=.12, color=PURPLE)
    except Exception:
        pass
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    return fig


def fig_prc(pipe, X_test, y_test):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    display = PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax,
                                                     color=CYAN, lw=2.2, alpha=.9)
    try:
        xd = display.line_.get_xdata()
        yd = display.line_.get_ydata()
        ax.fill_between(xd, yd, alpha=.12, color=CYAN)
    except Exception:
        pass
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    return fig


def fig_feature_importance(pipe, X_test, y_test, feature_names, clf_name, n_top=15):
    fig, ax = plt.subplots(figsize=(7, 5))

    if clf_name in TREE_BASED:
        importances = pipe.named_steps["clf"].feature_importances_
        idx   = np.argsort(importances)[-n_top:]
        vals  = importances[idx]
        names = [feature_names[i] for i in idx]
        title = f"Top {n_top} Feature Importances (Gini / Impurity)"
    else:
        result = permutation_importance(pipe, X_test, y_test,
                                        n_repeats=15, random_state=42, n_jobs=-1)
        idx   = np.argsort(result.importances_mean)[-n_top:]
        vals  = result.importances_mean[idx]
        errs  = result.importances_std[idx]
        names = [feature_names[i] for i in idx]
        title = f"Top {n_top} Permutation Importances (±1 std)"
        ax.barh(names, vals, xerr=errs, color=PURPLE, ecolor=SLATE,
                capsize=4, alpha=.85, edgecolor="white")
        ax.axvline(0, color=SLATE, lw=.8, linestyle="--", alpha=.5)
        ax.set_xlabel("Mean Accuracy Decrease")
        ax.set_title(title)
        plt.tight_layout()
        return fig

    palette = sns.color_palette("viridis_r", len(idx))
    ax.barh(names, vals, color=palette, alpha=.9, edgecolor="white")
    ax.axvline(0, color=SLATE, lw=.8, linestyle="--", alpha=.5)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def fig_learning_curve(pipe, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, tr_scores, cv_scores = learning_curve(
        pipe, X_train, y_train, cv=cv,
        train_sizes=np.linspace(.1, 1.0, 9),
        scoring="accuracy", n_jobs=-1,
    )
    tr_m, tr_s = tr_scores.mean(1), tr_scores.std(1)
    cv_m, cv_s = cv_scores.mean(1), cv_scores.std(1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr_m, "o-", color=PURPLE, lw=2.2, label="Training accuracy")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=.14, color=PURPLE)
    ax.plot(sizes, cv_m, "s-", color=CYAN, lw=2.2, label="CV accuracy")
    ax.fill_between(sizes, cv_m - cv_s, cv_m + cv_s, alpha=.14, color=CYAN)
    gap = tr_m[-1] - cv_m[-1]
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve  (Train-CV gap = {gap:.3f})")
    ax.legend(fontsize=9)
    ax.set_ylim(.5, 1.02)
    plt.tight_layout()
    return fig


def fig_cv_analysis(cv_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Histogram
    ax1.hist(cv_scores, bins=min(8, len(cv_scores)), color=PURPLE,
             edgecolor="white", alpha=.85)
    ax1.axvline(cv_scores.mean(), color=RED, lw=2, linestyle="--",
                label=f"μ = {cv_scores.mean():.4f}")
    ax1.axvline(cv_scores.mean() - cv_scores.std(), color=AMBER, lw=1.4,
                linestyle=":", label=f"μ±σ  (σ={cv_scores.std():.4f})")
    ax1.axvline(cv_scores.mean() + cv_scores.std(), color=AMBER, lw=1.4, linestyle=":")
    ax1.set_xlabel("CV Accuracy")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Score Distribution")
    ax1.legend(fontsize=8.5)

    # Per-fold bar
    colors = sns.color_palette("viridis", len(cv_scores))
    bars = ax2.bar(range(1, len(cv_scores)+1), cv_scores, color=colors, edgecolor="white")
    ax2.axhline(cv_scores.mean(), color=RED, lw=2, linestyle="--")
    for bar, s in zip(bars, cv_scores):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + .001, f"{s:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Fold Scores")
    ax2.set_xticks(range(1, len(cv_scores)+1))
    lo = max(0, cv_scores.min() - .04)
    ax2.set_ylim(lo, min(1.01, cv_scores.max() + .03))

    plt.tight_layout()
    return fig


def fig_class_dist(df_raw):
    counts = df_raw["type"].value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PURPLE, CYAN], edgecolor="white", linewidth=1.5, width=.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 25,
                f"{v:,}\n({v/counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Mushroom Type")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    return fig


def fig_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(13, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
                linewidths=.25, ax=ax, cbar_kws={"shrink": .75},
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix (Lower Triangle)")
    plt.tight_layout()
    return fig


def fig_top_feature_dists(df, feature_names, top_n=6):
    """Distribution of top-n most target-correlated features split by class."""
    corr_with_target = df[feature_names].corrwith(df["type"]).abs().nlargest(top_n)
    feats = corr_with_target.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, feat in zip(axes.flat, feats):
        for cls_val, label, color in zip([0, 1], CLASS_NAMES, [PURPLE, CYAN]):
            subset = df[df["type"] == cls_val][feat]
            ax.hist(subset, bins=20, alpha=.65, color=color, label=label, edgecolor="white")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Encoded Value")
        ax.legend(fontsize=7)
    plt.suptitle("Top Feature Distributions by Class", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def compare_all_models_cached(X_train_arr, X_test_arr, y_train_arr, y_test_arr, col_names):
    X_train = pd.DataFrame(X_train_arr, columns=col_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=col_names)
    y_train = pd.Series(y_train_arr)
    y_test  = pd.Series(y_test_arr)

    candidates = {
        "SVM":               build_pipeline("Support Vector Machine (SVM)",
                                            SVC(kernel="rbf", probability=True, random_state=42)),
        "Logistic Reg.":     build_pipeline("Logistic Regression",
                                            LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest":     build_pipeline("Random Forest",
                                            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        "KNN":               build_pipeline("K-Nearest Neighbors (KNN)",
                                            KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree":     build_pipeline("Decision Tree",
                                            DecisionTreeClassifier(max_depth=12, random_state=42)),
        "Gradient Boosting": build_pipeline("Gradient Boosting",
                                            GradientBoostingClassifier(n_estimators=100, random_state=42)),
        "Naive Bayes":       build_pipeline("Naive Bayes",
                                            GaussianNB()),
    }
    rows = []
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        m, _ = compute_metrics(pipe, X_test, y_test)
        m["Model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("Model")


def fig_model_comparison(df_cmp):
    keep_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    df_plot   = df_cmp[[c for c in keep_cols if c in df_cmp.columns and df_cmp[c].notna().all()]]

    n_metrics = len(df_plot.columns)
    n_models  = len(df_plot)
    x     = np.arange(n_models)
    width = 0.75 / n_metrics
    colors = sns.color_palette("tab10", n_metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, color) in enumerate(zip(df_plot.columns, colors)):
        offset = (i - n_metrics / 2) * width + width / 2
        bars   = ax.bar(x + offset, df_plot[col], width * .92, label=col,
                        color=color, alpha=.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + .001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("All Classifiers — Side-by-Side Metric Comparison")
    ax.set_ylim(max(0.0, df_plot.values.min() - .07), 1.06)
    ax.legend(loc="lower right", fontsize=9, ncol=n_metrics)
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOAD
# ═══════════════════════════════════════════════════════════════════════════════
df_enc, df_raw = load_data()
X_train, X_test, y_train, y_test = split_data(df_enc)
FEATURE_NAMES = X_train.columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 0.4rem;'>
        <div style='font-size:2.2rem;'>🤖</div>
        <div style='font-size:1.15rem;font-weight:800;color:#0f172a;letter-spacing:-0.01em;'>ML Classifier Pro</div>
        <div style='font-size:.75rem;color:#64748b;font-weight:500;margin-top:.15rem;'>Professional Visualizer v2.0</div>
    </div>
    <hr style='border-color:#e2e8f0;margin:.6rem 0 1rem;'>
    """, unsafe_allow_html=True)

    clf_name = st.selectbox("🧠 Classifier", [
        "Support Vector Machine (SVM)",
        "Logistic Regression",
        "Random Forest",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree",
        "Gradient Boosting",
        "Naive Bayes",
    ])

    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7c3aed;font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;'>⚙️ Hyperparameters</div>", unsafe_allow_html=True)

    # ── Build classifier ──────────────────────────────────────────────────────
    if clf_name == "Support Vector Machine (SVM)":
        C      = st.slider("C  (Regularization)", .01, 20., 1., .05,
                            help="Higher C → less regularisation, fits training data more tightly.")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"],
                               help="Kernel transforms the feature space.")
        gamma  = st.selectbox("Gamma", ["scale", "auto"],
                               help="Kernel coefficient (scale = 1/(n_features·Var(X))).")
        clf_obj = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)

    elif clf_name == "Logistic Regression":
        C        = st.slider("C  (Regularization)", .01, 20., 1., .05)
        max_iter = st.slider("Max Iterations", 100, 3000, 1000, 100)
        solver   = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
        clf_obj  = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)

    elif clf_name == "Random Forest":
        n_est    = st.slider("Number of Trees", 10, 500, 100, 10,
                              help="More trees → higher accuracy, slower training.")
        max_d    = st.slider("Max Depth", 1, 40, 15, 1)
        min_ss   = st.slider("Min Samples Split", 2, 20, 2, 1)
        max_feat = st.selectbox("Max Features / Split", ["sqrt", "log2"],
                                 help="sqrt is standard for classification.")
        clf_obj  = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                           min_samples_split=min_ss, max_features=max_feat,
                                           random_state=42, n_jobs=-1)

    elif clf_name == "K-Nearest Neighbors (KNN)":
        k       = st.slider("K  (Neighbours)", 1, 30, 5, 1)
        weights = st.selectbox("Weight Function", ["uniform", "distance"])
        metric  = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"])
        algo    = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree"])
        clf_obj = KNeighborsClassifier(n_neighbors=k, weights=weights,
                                        metric=metric, algorithm=algo)

    elif clf_name == "Decision Tree":
        max_d   = st.slider("Max Depth", 1, 30, 10, 1,
                             help="Deeper → risk of overfitting.")
        crit    = st.selectbox("Split Criterion", ["gini", "entropy", "log_loss"])
        min_ss  = st.slider("Min Samples Split", 2, 20, 2, 1)
        min_sl  = st.slider("Min Samples Leaf", 1, 20, 1, 1)
        ccp     = st.slider("CCP Alpha (Pruning)", 0.0, 0.05, 0.0, 0.001,
                             help="Larger α → more pruning to reduce overfitting.")
        clf_obj = DecisionTreeClassifier(max_depth=max_d, criterion=crit,
                                          min_samples_split=min_ss,
                                          min_samples_leaf=min_sl,
                                          ccp_alpha=ccp, random_state=42)

    elif clf_name == "Gradient Boosting":
        n_est  = st.slider("Boosting Stages", 50, 500, 100, 10)
        lr     = st.slider("Learning Rate", .005, .5, .1, .005,
                            help="Lower rate + more stages often gives best results.")
        max_d  = st.slider("Max Tree Depth", 1, 10, 3, 1)
        subs   = st.slider("Subsample Fraction", .4, 1.0, 1.0, .05,
                            help="< 1.0 adds stochastic element → often better generalisation.")
        clf_obj = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                              max_depth=max_d, subsample=subs,
                                              random_state=42)

    elif clf_name == "Naive Bayes":
        vs_exp  = st.slider("Var Smoothing  (1e-x)", 1, 15, 9, 1,
                             help="Larger → more smoothing, more bias.")
        clf_obj = GaussianNB(var_smoothing=10 ** (-vs_exp))

    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

    run_cv       = st.checkbox("📊 Run 5-Fold CV", value=True,
                                help="Stratified cross-validation on the full dataset.")
    train_btn    = st.button("🚀 Train & Evaluate")
    compare_btn  = st.button("🏆 Compare All Models")

    st.markdown("""
    <hr style='border-color:#e2e8f0;'>
    <div style='font-size:.72rem;color:#64748b;text-align:center;line-height:1.4;'>
        <strong>Mushroom UCI Dataset</strong><br>
        8,124 samples · 22 features<br>
        Stratified 70/30 train-test split · seed=42
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:1.8rem;font-weight:800;color:#18181b;margin-bottom:.1rem;'>
    🤖 ML Classification Visualizer <span style='color:#7c3aed;'>Pro</span>
</h1>
<p style='color:#64748b;font-size:.9rem;margin-top:0;'>
    Professional binary classification analysis — Mushroom Edibility Dataset
</p>
""", unsafe_allow_html=True)

tab_data, tab_res, tab_plots, tab_cv, tab_cmp = st.tabs([
    "📊 Dataset Explorer",
    "🎯 Model Results",
    "📈 Visualizations",
    "🔄 Cross-Validation",
    "🏆 Compare All Models",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",  f"{len(df_enc):,}")
    c2.metric("Features",       len(FEATURE_NAMES))
    c3.metric("Training Set",   f"{len(X_train):,}  ({len(X_train)/len(df_enc):.0%})")
    c4.metric("Test Set",       f"{len(X_test):,}   ({len(X_test)/len(df_enc):.0%})")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        section("Class Distribution")
        st.pyplot(fig_class_dist(df_raw))
    with col_b:
        section("Feature Correlation Matrix")
        st.pyplot(fig_corr_heatmap(df_enc))

    section("Top Feature Distributions by Class")
    st.pyplot(fig_top_feature_dists(df_enc, FEATURE_NAMES))

    section("Raw Dataset Preview  (first 20 rows)")
    st.dataframe(df_raw.head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
if train_btn:
    pipeline = build_pipeline(clf_name, clf_obj)
    with st.spinner(f"⏳ Training {clf_name} …"):
        pipeline.fit(X_train, y_train)
    metrics, y_pred = compute_metrics(pipeline, X_test, y_test)

    st.session_state["pipeline"]    = pipeline
    st.session_state["metrics"]     = metrics
    st.session_state["y_pred"]      = y_pred
    st.session_state["clf_name"]    = clf_name
    st.session_state["trained"]     = True


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_res:
    if st.session_state.get("trained"):
        m        = st.session_state["metrics"]
        y_pred   = st.session_state["y_pred"]
        clf_lbl  = st.session_state["clf_name"]

        badge(clf_lbl)

        section("Core Metrics")
        cols = st.columns(4)
        for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score"]):
            with col:
                metric_card(key, m[key])

        section("Advanced Metrics")
        cols2 = st.columns(4)
        adv = {
            "ROC-AUC":      (m["ROC-AUC"],      True),
            "Avg Precision": (m["Avg Precision"], True),
            "MCC (norm.)":   (m["MCC"],           True),
            "Cohen κ (norm.)":(m["Cohen κ"],      True),
        }
        for col, (label, (val, is_pct)) in zip(cols2, adv.items()):
            with col:
                if val is not None:
                    metric_card(label, val, is_pct=is_pct,
                                sub="Normalised to 0–1" if "norm" in label else "")
                else:
                    col.info(f"{label}: N/A")

        section("Full Classification Report")
        report = classification_report(y_pred=y_pred, y_true=y_test,
                                        target_names=CLASS_NAMES, output_dict=True)
        rpt_df = pd.DataFrame(report).transpose()
        st.dataframe(
            rpt_df.style.format("{:.4f}")
                  .background_gradient(subset=["precision","recall","f1-score"],
                                       cmap="RdYlGn", vmin=.5, vmax=1.0),
            use_container_width=True,
        )

        section("Prediction Export")
        pred_df = pd.DataFrame({
            "True Label":  [CLASS_NAMES[int(v)] for v in y_test],
            "Predicted":   [CLASS_NAMES[int(v)] for v in y_pred],
            "Correct":     ["✅" if a == b else "❌" for a, b in zip(y_test, y_pred)],
        })
        csv_bytes = pred_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Predictions CSV", csv_bytes,
                            "predictions.csv", "text/csv")
    else:
        st.markdown("""<div class='info-box'>
            👈 Set hyperparameters in the sidebar, then click <strong>🚀 Train & Evaluate</strong>.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_plots:
    if st.session_state.get("trained"):
        pipe    = st.session_state["pipeline"]
        clf_lbl = st.session_state["clf_name"]
        badge(clf_lbl)

        # ── Row 1: Confusion · ROC · PRC ──────────────────────────────────
        section("Diagnostic Curves")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(fig_confusion(pipe, X_test, y_test))
        with c2:
            try:
                st.pyplot(fig_roc(pipe, X_test, y_test))
            except Exception as e:
                st.warning(f"ROC Curve unavailable: {e}")
        with c3:
            try:
                st.pyplot(fig_prc(pipe, X_test, y_test))
            except Exception as e:
                st.warning(f"PR Curve unavailable: {e}")

        # ── Row 2: Feature Importance ──────────────────────────────────────
        section("Feature Importance")
        if clf_lbl not in TREE_BASED:
            st.markdown("""<div class='warn-box'>
                ⚠️ <strong>Permutation Importance</strong> is used for this model-class 
                (model-agnostic, but slower). Wait a moment…
            </div>""", unsafe_allow_html=True)
        with st.spinner("Computing feature importances …"):
            st.pyplot(fig_feature_importance(pipe, X_test, y_test, FEATURE_NAMES, clf_lbl))

        # ── Row 3: Learning Curve ──────────────────────────────────────────
        section("Learning Curve")
        with st.spinner("Computing learning curve …"):
            st.pyplot(fig_learning_curve(pipe, X_train, y_train))

        st.markdown("""<div class='info-box'>
            📌 <strong>Interpreting the Learning Curve:</strong>
            A large gap between training and CV accuracy indicates <em>high variance (overfitting)</em>.
            Both curves plateauing at a low value indicates <em>high bias (underfitting)</em>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='info-box'>
            Train a model first to view diagnostic plots.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cv:
    if st.session_state.get("trained"):
        pipe    = st.session_state["pipeline"]
        clf_lbl = st.session_state["clf_name"]
        badge(clf_lbl + " — 5-Fold Stratified Cross-Validation")

        if run_cv:
            X_full = pd.concat([X_train, X_test])
            y_full = pd.concat([y_train, y_test])
            cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            with st.spinner("Running 5-fold CV …"):
                scores = cross_val_score(pipe, X_full, y_full,
                                          cv=cv, scoring="accuracy", n_jobs=-1)

            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Mean CV Accuracy",  scores.mean())
            with c2: metric_card("Std Deviation",     scores.std(), is_pct=False)
            with c3: metric_card("Min Fold Score",    scores.min())
            with c4: metric_card("Max Fold Score",    scores.max())

            section("Score Analysis")
            st.pyplot(fig_cv_analysis(scores))

            section("Per-Fold Table")
            fold_df = pd.DataFrame({
                "Fold":        [f"Fold {i+1}" for i in range(len(scores))],
                "Accuracy":    scores,
                "Δ from Mean": scores - scores.mean(),
                "Status":      ["🟢 Above avg" if s >= scores.mean() else "🟡 Below avg"
                                for s in scores],
            })
            st.dataframe(
                fold_df.style.format({"Accuracy": "{:.4f}", "Δ from Mean": "{:+.4f}"}),
                use_container_width=True,
            )
        else:
            st.info("Enable **📊 Run 5-Fold CV** in the sidebar.")
    else:
        st.markdown("""<div class='info-box'>Train a model first to view cross-validation.</div>""",
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — COMPARE ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    if compare_btn or st.session_state.get("comparison_done"):
        if compare_btn:
            with st.spinner("⏳ Training & evaluating all 7 classifiers …"):
                cmp_df = compare_all_models_cached(
                    X_train.values, X_test.values,
                    y_train.values, y_test.values,
                    FEATURE_NAMES,
                )
            st.session_state["comparison_done"]    = True
            st.session_state["comparison_results"] = cmp_df
        else:
            cmp_df = st.session_state["comparison_results"]

        section("Leaderboard")
        num_cols = cmp_df.select_dtypes("number").columns.tolist()
        st.dataframe(
            cmp_df[num_cols].style
                .format("{:.4f}")
                .background_gradient(cmap="RdYlGn", vmin=.6, vmax=1.0)
                .highlight_max(color="#bbf7d0", axis=0)
                .highlight_min(color="#fee2e2", axis=0),
            use_container_width=True,
        )

        section("Visual Comparison")
        st.pyplot(fig_model_comparison(cmp_df))

        best = cmp_df["F1 Score"].idxmax()
        st.success(f"🏆 **Best Model (F1 Score):** {best} — "
                   f"F1 = {cmp_df.loc[best, 'F1 Score']:.4f}  |  "
                   f"Accuracy = {cmp_df.loc[best, 'Accuracy']:.4f}  |  "
                   f"ROC-AUC = {cmp_df.loc[best, 'ROC-AUC']:.4f}")

        csv_cmp = cmp_df.to_csv().encode()
        st.download_button("⬇️ Download Comparison CSV", csv_cmp,
                            "model_comparison.csv", "text/csv")
    else:
        st.markdown("""
        <div class='info-box'>
            <strong>📊 Automated Model Benchmarking</strong><br>
            Click <strong>🏆 Compare All Models</strong> in the sidebar to train and 
            evaluate all 7 classifiers with default hyperparameters side-by-side.
            Results are cached so subsequent views are instant.
        </div>""", unsafe_allow_html=True)
