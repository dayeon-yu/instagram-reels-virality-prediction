"""평가·시각화 유틸 — 모델 비교표, ROC 곡선 겹쳐 그리기, 신호 유무 진단."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]


def score_table(y_true, probs: dict, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for name, p in probs.items():
        pred = (np.asarray(p) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"model": name,
                     "Accuracy": accuracy_score(y_true, pred),
                     "Precision": precision_score(y_true, pred, zero_division=0),
                     "Recall": recall_score(y_true, pred, zero_division=0),
                     "F1": f1_score(y_true, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_true, p),
                     "pred_viral_rate": pred.mean(), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return pd.DataFrame(rows).set_index("model")


def plot_roc(y_true, probs: dict, title: str, path: str):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for (name, p), c in zip(probs.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=1.8, color=c, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def signal_diagnostics(X_train, y_train, X_test, y_test, random_state: int = 42) -> pd.DataFrame:
    """남은 피처에 학습 가능한 신호가 있는지 진단한다.

    1) 피처별 ROC-AUC (단변량, 방향 무시) 와 mutual information
    2) 라벨을 무작위로 섞은 뒤 같은 모델을 학습했을 때의 AUC (영가설 기준선)
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = []
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    for c, m in zip(X_train.columns, mi):
        auc = roc_auc_score(y_test, X_test[c]); auc = max(auc, 1 - auc)
        rows.append({"feature": c, "univariate_AUC": auc, "mutual_info": m})
    diag = pd.DataFrame(rows).sort_values("univariate_AUC", ascending=False).reset_index(drop=True)

    gb = HistGradientBoostingClassifier(max_iter=200, random_state=random_state)
    auc_real = roc_auc_score(y_test, gb.fit(X_train, y_train).predict_proba(X_test)[:, 1])
    rng = np.random.default_rng(random_state)
    auc_shuffled = roc_auc_score(y_test, gb.fit(X_train, rng.permutation(y_train)).predict_proba(X_test)[:, 1])
    diag.attrs["gbdt_auc_real"] = auc_real
    diag.attrs["gbdt_auc_shuffled_labels"] = auc_shuffled
    return diag
