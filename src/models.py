"""
5개 분류 모델 — 발표에서 쓴 **최종 설정**을 그대로 코드로 고정한다.
(LR: GridSearch 결과 / DT: GridSearch 결과 / RF·AdaBoost·XGBoost: 노트북의 최종 설정)
XGBoost가 설치돼 있지 않으면 같은 자리에 scikit-learn의 HistGradientBoosting을 넣고 이름에 표시한다.
"""
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
import sklearn

RANDOM_STATE = 42


def _l1_logreg(**kw):
    """L1 로지스틱 회귀 — scikit-learn 1.8부터 penalty= 가 deprecated 라 버전에 따라 인자를 바꾼다."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", **kw)
    return LogisticRegression(penalty="l1", solver="liblinear", **kw)


def get_models(y_train=None, random_state: int = RANDOM_STATE) -> dict:
    """이름 → (estimator, fit_kwargs). AdaBoost는 class_weight가 없어 sample_weight로 불균형을 보정한다."""
    models = {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", _l1_logreg(C=0.1, class_weight="balanced", max_iter=1000, random_state=random_state))]),
            {}),
        "Decision Tree": (
            DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=16,
                                   class_weight="balanced", random_state=random_state),
            {}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=random_state),
            {}),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
                               n_estimators=50, learning_rate=1.0, random_state=random_state),
            {"sample_weight": compute_sample_weight("balanced", y_train)} if y_train is not None else {}),
    }
    try:
        from xgboost import XGBClassifier
        spw = float((y_train == 0).sum() / max(1, (y_train == 1).sum())) if y_train is not None else 4.0
        models["XGBoost"] = (
            XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=random_state, n_jobs=-1),
            {})
    except ImportError:
        models["XGBoost (대체: HistGradientBoosting)"] = (
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=5,
                                           class_weight="balanced", random_state=random_state),
            {})
    return models
