# Explore-Mental-Health-Depression
Cuộc thi **[Kaggle Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11)** — phân loại nhị phân dự đoán trầm cảm (`Depression`: 0/1) từ dữ liệu khảo sát sức khỏe tâm thần.

---

## Pipeline tổng quan

```
Data (train + original)
        │
        ▼
┌───────────────────────────────────────┐
│         Preprocessing                │
│  p1/p3: process_native               │
│  p2/p4: process_detailed_impute      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Base Models (Tầng 1)         │
│  XGBoost · LightGBM (gbdt/goss/dart) │
│  CatBoost · Logistic Regression      │
│  AutoGluon (p1, p2)                  │
│  → 5-Fold Stratified CV → OOF preds  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Pseudo-Labeling              │
│  Chọn model "thầy" (OOF cao nhất)    │
│  Gán nhãn giả cho tập test           │
│  AutoGluon retrain trên              │
│  train + original + pseudo (p3, p4)  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Stacking (Tầng 2)            │
│  OOF matrix → LR + Ridge meta-model  │
│  Blend weight + threshold tối ưu     │
│  bằng Optuna (500 trials, TPE)       │
└───────────────────────────────────────┘
        │
        ▼
   submission.csv
```

---

## Preprocessing

Hai pipeline xử lý dữ liệu được dùng song song để tạo đa dạng cho ensemble:

| Pipeline | Hàm | Mô tả |
|---|---|---|
| `p1`, `p3` | `process_native` | Chỉ chuẩn hoá dtype, đồng bộ categories giữa train/test/original |
| `p2`, `p4` | `process_detailed_impute` | Ordinal encoding (sleep, diet, gender...) + conditional imputation theo nhóm Student/Professional |

> CatBoost bỏ qua `p1` do không xử lý được NaN thô — chỉ train trên `p2`.

---

## Base Models

| Model | Pipeline |
|---|---|
| XGBoost | p1, p2 |
| LightGBM (gbdt, goss, dart) | p1, p2 |
| CatBoost | p2 |
| Logistic Regression | p1, p2 |
| AutoGluon (`high_quality`, 1800s) | p1, p2 |

Tất cả base model dùng **5-fold Stratified Cross-Validation**, sinh ra OOF predictions và test predictions trung bình qua các fold.

---

## Pseudo-Labeling

1. Chọn model "thầy" có OOF accuracy cao nhất.
2. Dùng test predictions của "thầy" + threshold tối ưu để tạo nhãn giả cho tập test.
3. AutoGluon được train lại trên `train + original + pseudo` qua pipeline `p3` và `p4`.

---

## Stacking & Blending

- **Tầng 2:** OOF matrix của toàn bộ base model được dùng để train `Logistic Regression` và `Ridge Regression` meta-learner (qua cross-validation).
- **Blending:** Optuna (500 trials, TPE sampler) tối ưu đồng thời trọng số blend giữa LR và Ridge, và classification threshold để đạt accuracy cao nhất trên OOF.

---

## Cách chạy

Notebook thiết kế chạy trên **Kaggle**, cần mount các dataset sau:

```
playground-series-s4e11        # train.csv, test.csv, sample_submission.csv
depression-surveydataset-for-analysis  # final_depression_dataset_1.csv (original data)
oof-files                      # checkpoint.pkl (OOF từ lần chạy trước, tuỳ chọn)
```

Chạy toàn bộ cell theo thứ tự. Output:

```
submissions/sub_stacked model prediction.csv   # File nộp cuối
inf/model_predictions_checkpoint.pkl          # Checkpoint OOF để tái sử dụng
```
