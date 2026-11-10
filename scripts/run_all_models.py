#!/usr/bin/env python
"""
5개 모델을 **같은 전처리·같은 분할**로 한 번에 학습·평가하고 results/ 에 표와 그림을 남긴다.

사용법
    python scripts/run_all_models.py --data data/instagram_data.csv
    python scripts/run_all_models.py --data data/instagram_data.csv --thresholds 80 60 --leakage-upper-bound --diagnostics

출력 (results/)
    model_comparison_th{T}.csv          threshold T 에서의 5개 모델 지표
    figures/roc_th{T}.png               ROC 곡선 겹쳐 그리기
    threshold_experiment.csv            threshold 80 → 60 변화 (--thresholds 80 60 일 때)
    leakage_upper_bound.csv             누수 변수를 '포함'했을 때의 성능 = 제거한 정보의 크기 (--leakage-upper-bound)
    signal_diagnostics.csv              피처별 단변량 AUC·mutual information, 라벨 섞기 기준선 (--diagnostics)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import common_preprocess, load_raw  # noqa: E402
from src.models import get_models  # noqa: E402
from src.evaluate import score_table, plot_roc, signal_diagnostics  # noqa: E402


def run(df, threshold, out, *, keep_leakage=False, tag=""):
    X_tr, X_te, y_tr, y_te = common_preprocess(df, threshold=threshold, keep_leakage=keep_leakage)
    probs = {}
    for name, (model, fit_kw) in get_models(y_tr).items():
        model.fit(X_tr, y_tr, **fit_kw)
        probs[name] = model.predict_proba(X_te)[:, 1]
        print(f"  [{tag or f'th={threshold}'}] {name:<40s} AUC={roc_auc_score(y_te, probs[name]):.4f}")
    table = score_table(y_te, probs)
    plot_roc(y_te, probs, f"ROC — threshold {threshold}{' (+leakage features)' if keep_leakage else ''}",
             out / "figures" / f"roc_th{threshold}{'_leakage' if keep_leakage else ''}.png")
    return table, (X_tr, X_te, y_tr, y_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "instagram_data.csv"))
    ap.add_argument("--thresholds", nargs="+", type=int, default=[80])
    ap.add_argument("--leakage-upper-bound", action="store_true", help="누수 변수를 포함해 학습한 상한선 실험")
    ap.add_argument("--diagnostics", action="store_true", help="피처 신호 진단(단변량 AUC·MI·라벨 섞기)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load_raw(args.data)
    print(f"data: {df.shape}  viral(>=80) rate = {(df['virality_score'] >= 80).mean():.3f}")

    tables = {}
    for t in args.thresholds:
        tables[t], split = run(df, t, out)
        tables[t].round(4).to_csv(out / f"model_comparison_th{t}.csv")
        print(f"\n=== threshold {t} ===\n{tables[t][['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'pred_viral_rate']].round(4)}\n")

    if len(args.thresholds) >= 2:
        a, b = args.thresholds[:2]
        exp = pd.DataFrame({f"F1@{a}": tables[a]["F1"], f"F1@{b}": tables[b]["F1"],
                            f"AUC@{a}": tables[a]["ROC-AUC"], f"AUC@{b}": tables[b]["ROC-AUC"]})
        exp[f"ΔF1 ({a}→{b})"] = exp[f"F1@{b}"] - exp[f"F1@{a}"]
        exp[f"ΔAUC ({a}→{b})"] = exp[f"AUC@{b}"] - exp[f"AUC@{a}"]
        exp.round(4).to_csv(out / "threshold_experiment.csv")
        print(f"=== threshold experiment {a} → {b} ===\n{exp.round(4)}\n")

    if args.leakage_upper_bound:
        t = args.thresholds[0]
        ub, _ = run(df, t, out, keep_leakage=True, tag=f"th={t}+leakage")
        ub.round(4).to_csv(out / "leakage_upper_bound.csv")
        print(f"=== upper bound with leakage features (threshold {t}) ===\n{ub[['Accuracy', 'F1', 'ROC-AUC']].round(4)}\n")

    if args.diagnostics:
        X_tr, X_te, y_tr, y_te = split
        diag = signal_diagnostics(X_tr, y_tr, X_te, y_te)
        diag.round(4).to_csv(out / "signal_diagnostics.csv", index=False)
        print("=== signal diagnostics (threshold %d) ===" % args.thresholds[-1])
        print(diag.round(4).to_string(index=False))
        print(f"GBDT AUC — real labels {diag.attrs['gbdt_auc_real']:.4f} | shuffled labels {diag.attrs['gbdt_auc_shuffled_labels']:.4f}")


if __name__ == "__main__":
    main()
