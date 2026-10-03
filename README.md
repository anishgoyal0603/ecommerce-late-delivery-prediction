# E-commerce Late Delivery Prediction

Data science project done as part of the **Virtual Data Science Explorer Internship (Yuva Intern)**, July–August 2026.

The idea is to predict whether an online order will be delivered **after the estimated delivery date**, using order, product, seller and customer details. If late orders can be flagged early, the platform can warn customers, prioritise shipments or show more realistic delivery dates.

## Progress

| Week | Task | Status |
|------|------|--------|
| 1 | Project planning and strategy design | ✅ Done |
| 2 | Data cleaning and exploratory data analysis | ⏳ Upcoming |
| 3 | Feature engineering and model building | ⏳ Upcoming |
| 4 | Evaluation, insights and final report | ⏳ Upcoming |

## Week 1 – Project Plan

The full plan is in [`docs/Week1_Project_Plan_Anish_Goyal.docx`](docs/Week1_Project_Plan_Anish_Goyal.docx). It covers:

- Background, motivation and problem statement
- Objectives and scope (including what is out of scope)
- Proposed dataset – [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- Methodology based on CRISP-DM: collection → cleaning → EDA → feature engineering → modelling → evaluation
- Timeline of about 32 hours across 4 weeks
- Tools and Python libraries
- Expected outcomes, benefits, challenges and how to handle them

### Diagrams

The planning diagrams were made in Python with Matplotlib (`diagrams.py`).

**Project workflow**

![Workflow](images/fig1_workflow.png)

**Data pipeline**

![Pipeline](images/fig2_pipeline.png)

**Timeline**

![Timeline](images/fig3_timeline.png)

To regenerate them:

```bash
pip install matplotlib
python diagrams.py
```

## Problem setup (short version)

- **Type:** Binary classification
- **Target:** `is_late = 1` if actual delivery date > estimated delivery date, else `0`
- **Models planned:** Logistic Regression (baseline), Random Forest, XGBoost / LightGBM
- **Main metrics:** Recall and F1 for the late class, ROC-AUC, PR-AUC (accuracy alone is misleading because late orders are a small share)
- **Split:** by purchase date, not random, to mimic predicting future orders
- **Leakage check:** review scores are given after delivery, so they are only used in EDA, not as features

## Repository structure

```
├── docs/        # weekly reports
├── images/      # diagrams and charts
├── diagrams.py  # script used to make the Week 1 diagrams
├── requirements.txt
└── README.md
```

## Author

**Anish Goyal** – [github.com/anishgoyal0603](https://github.com/anishgoyal0603)
