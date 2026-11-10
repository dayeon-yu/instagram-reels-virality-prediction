"""
공통 전처리 파이프라인 — 5개 모델이 **동일한** 데이터·분할·시드로 비교되도록 한 곳에 모았다.
(노트북마다 복사돼 있던 1~16번 셀을 함수로 옮긴 것. 로직은 같고, 결측 대체만 train 통계로 fit하도록 고쳤다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# 업로드 *이후*에 생기는 성과 지표 → 예측 시점에 알 수 없음 (data leakage)
LEAKAGE_COLS = ["likes", "comments", "shares", "saves", "impressions", "reach", "engagement_rate"]
# 행마다 고유한 식별자 → 정보 없음 + One-Hot 시 메모리 폭증
ID_COLS = ["reel_id", "creator_id"]
SCORE_COL = "virality_score"


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def make_target(df: pd.DataFrame, threshold: int = 80) -> pd.Series:
    """virality_score >= threshold → Viral(1). 기본 80 (Viral ≈ 20%), 추가 실험은 60 (≈ 40%)."""
    return (df[SCORE_COL] >= threshold).astype(int)


def select_features(df: pd.DataFrame, *, keep_leakage: bool = False) -> pd.DataFrame:
    """누수 변수·ID·타겟 원본을 제거한 입력 행렬. keep_leakage=True 는 '상한선' 실험용."""
    drop = ID_COLS + [SCORE_COL] + ([] if keep_leakage else LEAKAGE_COLS)
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    for c in X.select_dtypes(include="bool").columns:   # 퍼셉트론/스케일러 입력을 위해 bool → 0/1
        X[c] = X[c].astype(int)
    return X


def common_preprocess(df: pd.DataFrame, *, threshold: int = 80, test_size: float = 0.2,
                      random_state: int = RANDOM_STATE, keep_leakage: bool = False):
    """팀 공통 전처리: 타겟 생성 → 누수/ID 제거 → stratified split → median imputation(train fit) → One-Hot(해당 시).

    Returns
    -------
    X_train, X_test, y_train, y_test  (pandas 객체)
    """
    y = make_target(df, threshold)
    X = select_features(df, keep_leakage=keep_leakage)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    imputer = SimpleImputer(strategy="median").fit(X_train[num_cols])      # train 통계로만 fit
    X_train[num_cols] = imputer.transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    if cat_cols:  # ID 제거 후에는 범주형이 남지 않지만, 다른 데이터에도 쓸 수 있게 남겨 둔다
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test, y_train, y_test
