
# Credit Card Fraud Detection

Binary classification project on a highly imbalanced dataset (fraud is only 0.17% of transactions). Built with scikit-learn and imbalanced-learn.

## Results

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|-------|--------|-----------|-----|---------|--------|
| Logistic Regression (no resampling) | 0.64 | 0.82 | 0.72 | 0.958 | 0.740 |
| Logistic Regression + SMOTE | 0.92 | 0.06 | 0.11 | 0.971 | 0.724 |
| Logistic Regression (class_weight=balanced) | 0.92 | 0.06 | 0.11 | 0.972 | 0.719 |
| Random Forest + SMOTE | 0.82 | 0.88 | 0.85 | 0.963 | 0.868 |

Random Forest trained on SMOTE-resampled data performed best overall, with the highest F1 and PR-AUC of any model tested.

One result worth calling out: SMOTE improved recall a lot for Logistic Regression (0.64 to 0.92) but destroyed precision (0.82 to 0.06). Random Forest didn't have this problem on the same resampled data. I think this comes down to how the two models make decisions: Logistic Regression fits one linear boundary across the whole feature space, so rebalancing the training data to 50/50 shifts that boundary hard, since the model is essentially told fraud is far more common than it really is. Random Forest builds many trees that each split on local feature thresholds, so it doesn't rely on one global boundary the same way, and handles the rebalanced data more gracefully.

## Why accuracy doesn't work here

Before building any real model, I checked what a classifier that just guesses "not fraud" every time would score:

```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
```

Accuracy: 99.83%
Fraud recall: 0.00%
F1 (fraud class): 0.00

With only 492 fraud cases out of 284,807 transactions, a model that ignores fraud completely still looks like it's performing well on accuracy alone. That's why this project evaluates models using precision, recall, F1, and PR-AUC instead, metrics that actually reflect performance on the minority class.

## Dataset

**Source**: [Kaggle, Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions from European cardholders (September 2013)
- 492 fraud cases, 0.172% of all transactions
- Features V1-V28: PCA-transformed (anonymized for privacy)
- Features: Time (seconds since first transaction), Amount (EUR)
- Target: Class (0 = legitimate, 1 = fraud)

Note: Since V1-V28 are PCA components, individual features can't be interpreted directly (they don't correspond to real-world variables like merchant category or location). The focus of this project is methodology, comparing resampling strategies and models on severely imbalanced data, rather than feature engineering.

## Project Structure

This project is contained in a single notebook: `Fraud_detection.ipynb`. It covers, in order: data preprocessing, exploratory analysis, class imbalance handling (undersampling, SMOTE, SMOTETomek), model training (Logistic Regression, Random Forest), evaluation, and threshold tuning. Generated plots are saved to the `figures/` folder.

```
credit-card-fraud-detection/
├── Fraud_detection.ipynb    # full analysis, start to finish
├── figures/                  # saved plots
│   ├── resampling_comparison.png
│   ├── feature_importance.png
│   ├── roc_pr_curves.png
│   └── threshold_analysis.png
└── README.md
```

Note: the dataset (`creditcard.csv`) isn't included in this repo due to file size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same directory to run the notebook.

## Methods

### 1. Preprocessing
- `RobustScaler` applied to `Amount` and `Time` (resistant to outliers, uses median and IQR instead of mean and standard deviation)
- V1-V28 left as-is, already PCA-normalized
- Stratified train/test split (80/20), preserving the 0.172% fraud ratio in both sets

### 2. Handling Class Imbalance

Three resampling strategies were compared on the training data:

| Strategy | Training Set Size | Fraud Samples |
|----------|-------------------|----------------|
| No resampling (baseline) | 227,845 | 394 (0.17%) |
| Random Undersampling | 788 | 394 (50%) |
| SMOTE | 454,902 | 227,451 (50%) |
| SMOTETomek | 454,902 | 227,451 (50%) |

Note: SMOTETomek produced the same class counts as SMOTE on this dataset, meaning no Tomek links (overlapping same-neighbor pairs) were found to remove after oversampling. This suggests the SMOTE-generated synthetic fraud points didn't create ambiguous overlap with the legitimate class in this case.

All resampling was applied only to the training data. The test set was kept at its original, imbalanced distribution throughout, so evaluation results reflect real-world class proportions.

### 3. Models
- Logistic Regression, tested as a baseline (no resampling), on SMOTE-resampled data, on undersampled data, and with `class_weight='balanced'`
- Random Forest (100 estimators, `class_weight='balanced'`), trained on SMOTE-resampled data

### 4. Evaluation

Primary metrics: Precision, Recall, F1 (fraud class), PR-AUC
Secondary metrics: ROC-AUC, confusion matrix

ROC-AUC was not used as the primary metric, since it can look misleadingly strong on imbalanced data. Precision-Recall AUC is more informative here because it's sensitive to how the model performs specifically on the rare fraud class.

Threshold tuning: after selecting the best model (Random Forest + SMOTE), the decision threshold was tuned instead of using the default 0.5 cutoff, to explore the tradeoff between catching more fraud and reducing false alarms.
