---
layout: post
title: "5개 모델이 전부 실패했다 — Instagram Reels 바이럴 예측 프로젝트 회고"
date: 2026-06-17 12:00:00 +0900
categories: [Machine Learning, Project]
tags: [Classification, EDA, Data Leakage, Class Imbalance, XGBoost, Data Science]
---

> Data Science 최종 팀프로젝트(Team 6) 회고.
> "Instagram Reels가 바이럴될지 예측하자"로 시작해, **다섯 모델이 모두 실패한 이유**를 데이터에서 찾은 이야기.

## 🎯 시작

- **주제**: Reel의 메타데이터로 **바이럴 여부(viral/non-viral)** 예측
- **데이터**: Kaggle *What Makes an Instagram Reel Go Viral* — 100,000개 × 26개 변수
- **타겟**: `virality_score >= 80`이면 Viral (전체의 약 20%)
- **계획**: 여러 분류 모델을 붙여보고 가장 좋은 걸 고르자 — 흔한 접근

## 🧹 공정한 비교를 위한 공통 전처리

여러 모델을 비교하려면 **전처리를 통일**해야 함. 안 그러면 성능 차이가 모델 때문인지 전처리 때문인지 알 수 없음.

- **Data Leakage 변수 제거**: `likes`, `comments`, `shares`, `saves`, `impressions`, `reach`, `engagement_rate`
  → 업로드 *이후* 생기는 지표라 예측 시점엔 알 수 없음
- **ID 제거**: `reel_id`, `creator_id` (각 10만 개 고유값 → 정보 없음 + 메모리 폭증)
- **결측치**: 중앙값 대체 / **동일 split · 동일 seed(42)** 통일

## 🤖 다섯 모델, 하나의 결과

| 모델 | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.5026 | 0.2896 | 0.4997 |
| Decision Tree | 0.6767 | 0.2076 | 0.5008 |
| AdaBoost | 0.2026 | 0.3365 | 0.4999 |
| Random Forest | 0.7976 | ≈ 0.00 | ≈ 0.50 |
| XGBoost | Low | Low | 0.5110 |

각 모델은 **저마다 다른 방식으로 실패**했음.

- Logistic Regression → 선형 신호가 없음
- AdaBoost → 거의 전부 Viral로 예측 (false positive 폭발)
- Random Forest → 전부 Non-Viral로 예측 (다수 클래스로 도망)
- XGBoost → 강력한데도 경계를 못 찾음

그런데 공통점이 하나. **ROC-AUC가 전부 0.5 근처** — 사실상 동전 던지기.

> XGBoost까지 실패했다면, 문제는 모델이 아니라 **데이터**일 가능성이 큼.

## 🔬 "불균형 탓 아닐까?" — 임계값 실험

가설을 세워봄: Viral이 20%밖에 안 되니 불균형이 문제 아닐까?

- `virality_score` 임계값을 **80 → 60**으로 낮춰 Viral 비율을 20% → 40%로
- **결과**: F1은 올랐지만 **ROC-AUC는 여전히 0.5**
- 균형을 맞춰도 **판별력 자체는 그대로** → 불균형이 근본 원인은 아니었음

## 📊 진짜 원인 — Advanced EDA

데이터를 다시 뜯어보니 실패는 예정되어 있었음.

- **팔로워 수 ≠ 바이럴**: 팔로워가 많아도 평균 Virality Score는 비슷
- **콘텐츠 특성**(길이·해시태그): 그룹 간 차이 미미
- **Retention**: Viral이 조금 높지만 분포가 심하게 겹침
- **상관관계**: Virality와 강하게 연결된 변수가 **하나도 없음**

원인을 3가지로 정리:

1. **심한 클래스 불균형** (20 : 80)
2. **클래스 간 분포 중첩** (Viral/Non-Viral의 feature 범위가 거의 같음)
3. **낮은 변수 분리력** (decision boundary가 안 그려짐)

### 그리고 가장 아이러니한 지점

**Data Leakage를 막으려고 제거한 변수들**(likes, comments, shares, reach 등)이
사실은 **바이럴을 가장 잘 설명하는 신호**였음.

→ 정당하게 누수를 제거한 순간, 예측력의 핵심도 함께 사라진 것.

## 💡 배운 점

- **좋은 프로젝트 = 높은 점수가 아니다** — *왜 안 되는지*를 규명하는 것도 결과
- **Data Leakage 제거는 양날의 검** — 정보 손실까지 함께 고려해야 함
- **모델을 늘리기 전에 데이터를 의심하라** — 5개를 돌리고 나서야 데이터를 본 게 아쉬움
- 메타데이터만으로 설명 안 되는 문제엔, **콘텐츠 자체(영상·주제·감정)** 정보가 필요

> **The limitation was not the model. The limitation was the dataset.**

---

*코드·발표자료 전문은 [GitHub 저장소](https://github.com/dayeon-yu/instagram-reels-virality-prediction)에서 확인할 수 있습니다.*
