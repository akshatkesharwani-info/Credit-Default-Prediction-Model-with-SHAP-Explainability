# Credit Default Prediction Model with SHAP Explainability

ML Engineer Track | Difficulty: Medium-Hard**

100% free and open-source tools only — no paid APIs, no credit card, runs on Google Colab's free tier.

---

## 📌 Overview

Lenders need not just an accurate default prediction but a transparent, explainable one for regulatory and credit-committee purposes. Build an XGBoost credit default classifier and use SHAP to explain both global feature importance and individual loan-level decisions -- entirely with free, open-source libraries.

**Domain:** Credit Risk (JPMorgan-style)

---

## 📊 Dataset

Credit Risk Dataset -- ONE single CSV file (credit_risk_dataset.csv), no accepted/rejected file confusion (auto-generates sample data if file is missing)

**Source:** [https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

> This script also includes a **safe fallback**: if the real dataset file isn't found next to the
> notebook/script, it automatically generates a small realistic sample dataset with the same column
> names, so the whole pipeline still runs end-to-end even before you've downloaded the real data.

---

## 🛠️ Tech Stack

Python 3 | XGBoost | SHAP | scikit-learn | pandas

**Skills demonstrated:** Python, scikit-learn, XGBoost, SHAP, pandas

---

## 🎯 What This Project Builds

- A cleaned feature set from loan-level data (income, employment length, loan amount, credit history length)
- A train/test split with proper class-imbalance handling (defaults are the minority class)
- An XGBoost classifier tuned for AUC/recall on the minority (default) class
- Global SHAP summary plot showing which features drive default risk most
- Local SHAP waterfall plots explaining individual loan decisions
- A one-page explainability report per borrower ready for a credit committee

---

## 🧭 Step-by-Step Approach

### Step 1: Load Data (with a Safe Fallback)

**What:** Load the single credit_risk_dataset.csv file, or auto-build a realistic sample dataset if it isn't found

**Why:** This dataset is just ONE csv file (unlike some Kaggle credit datasets that ship accepted+rejected as two separate files) -- much simpler to work with, and a missing file still shouldn't stop the script from running

**How:** if os.path.exists(path): pd.read_csv(path) else: generate synthetic loan rows with numpy


### Step 2: Handle Class Imbalance

**What:** Use scale_pos_weight since defaults are a minority class

**Why:** Without imbalance handling, the model will just predict 'no default' for everyone

**How:** scale_pos_weight = (negative_count / positive_count) passed into XGBClassifier


### Step 3: Train XGBoost Classifier

**What:** Fit an XGBoost model and evaluate with AUC, precision/recall, and a confusion matrix

**Why:** AUC and recall on the default class matter more than raw accuracy for credit risk

**How:** XGBClassifier(scale_pos_weight=..., eval_metric='auc').fit(X_train, y_train)


### Step 4: Explain with SHAP

**What:** Compute SHAP values and plot global importance + individual waterfalls

**Why:** SHAP is the industry-standard explainability tool for regulator-facing credit models

**How:** shap.TreeExplainer(model).shap_values(X_test); shap.summary_plot(...)


---

## 📈 Dashboard / Reporting Ideas

- KPI cards: AUC, precision, recall, F1 for the current model version
- SHAP summary plot embedded as the headline 'what drives default risk' visual
- Per-borrower explanation panel: paste in a loan ID, show its SHAP waterfall
- Table: highest-risk current applications sorted by predicted default probability
- Confusion matrix heatmap for quick model-health monitoring

---

## 💡 Key Insights

- This dataset is a single clean CSV (credit_risk_dataset.csv) -- much easier to work with than lending datasets split across separate accepted/rejected files
- loan_percent_income and credit history length are consistently top SHAP features in this dataset
- scale_pos_weight is a simple, effective first step for imbalanced credit default data before reaching for SMOTE
- SHAP waterfall plots are what let a credit committee understand and defend an individual denial decision
- This full pipeline (XGBoost + SHAP) runs on 100% free, open-source Python libraries with no cloud cost

---

## 🚀 How to Run

1. Open the `.py` file in **Google Colab** (free tier — no GPU or paid compute needed) or run it locally with Python 3.
2. Install dependencies with the `pip install ...` line at the top of the script (all free, open-source packages).
3. (Optional) Download the real dataset from the Kaggle link above and place it in the same folder — the filename the script expects is noted in the code's data-loading step. If you skip this, the script auto-generates sample data so you can still see it run.
4. Run the script top to bottom. Outputs (charts, CSVs, model files) are saved in the working directory.

```bash
pip install -r requirements.txt   # or the pip install line at the top of the script
python MLEng_01_Credit_Default_Prediction_SHAP.py
```

---

## 📂 Repo Structure

```
credit-default-prediction-model-shap-explainability/
├── MLEng_01_Credit_Default_Prediction_SHAP.py       # complete, runnable, free-only solution code
├── README.md              # this file
└── outputs/                # charts, CSVs, and model files generated on run
```

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes to demonstrate applied ML/quant-risk
skills. It is not financial, credit, or investment advice, and should not be used for real lending,
trading, or compliance decisions without proper review by a licensed professional.

---

*Part of a 20-project AI Engineer + ML Engineer portfolio focused on finance and consulting use cases —
built entirely with free, open-source tools.*
