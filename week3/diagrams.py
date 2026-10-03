"""Diagrams for the Week 3 ML plan."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "images/"


def box(ax, x, y, w, h, t, c, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=c, ec="#333", lw=1.2))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs)


def arr(ax, a, b, ls="-", color="#333", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# Fig 1 - end to end ML workflow
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 24); ax.set_ylim(0, 12.4); ax.axis("off")
row1 = [(2.3, "Problem\ndefinition\n(is_late?)", "#FFF2B3"),
        (6.7, "Cleaned data\n(from Week 2\nEDA)", "#D9EAD3"),
        (11.1, "Time-based split\nTrain | Valid | Test", "#CFE2F3"),
        (15.5, "Preprocessing\npipeline\n(fit on train only)", "#D9D2E9"),
        (19.9, "Feature\nselection", "#EAD1DC")]
for x, t, c in row1:
    box(ax, x, 9.8, 3.7, 2.4, t, c)
for a, b in zip(row1[:-1], row1[1:]):
    arr(ax, (a[0] + 1.85, 9.8), (b[0] - 1.85, 9.8))
row2 = [(19.9, "Baseline models\n(Dummy, Logistic\nRegression)", "#F4CCCC"),
        (15.5, "Advanced models\n(Decision Tree,\nRandom Forest,\nXGBoost/LightGBM)", "#F4CCCC"),
        (11.1, "Hyperparameter\ntuning with\ntime-series CV", "#FCE5CD"),
        (6.7, "Threshold\ntuning +\nmodel choice", "#FCE5CD"),
        (2.3, "Final test on\nhold-out set\n(used once)", "#D0E0E3")]
for x, t, c in row2:
    box(ax, x, 5.6, 3.7, 2.6, t, c)
arr(ax, (19.9, 8.6), (19.9, 6.9))
for a, b in zip(row2[:-1], row2[1:]):
    arr(ax, (a[0] - 1.85, 5.6), (b[0] + 1.85, 5.6))
box(ax, 2.3, 1.6, 3.7, 2.2, "Explain results\n(feature importance,\nSHAP)", "#D9EAD3")
box(ax, 6.7, 1.6, 3.7, 2.2, "Save pipeline\n(joblib)", "#CFE2F3")
box(ax, 11.1, 1.6, 3.7, 2.2, "Deploy as API /\nbatch scoring\n(optional)", "#D9D2E9")
box(ax, 15.5, 1.6, 3.7, 2.2, "Monitor drift &\nperformance", "#FFF2B3")
arr(ax, (2.3, 4.3), (2.3, 2.7))
for x1, x2 in [(2.3, 6.7), (6.7, 11.1), (11.1, 15.5)]:
    arr(ax, (x1 + 1.85, 1.6), (x2 - 1.85, 1.6))
arr(ax, (17.35, 1.6), (21.5, 4.3), ls="--", color="#C0392B", rad=0.3)
ax.text(21.9, 2.0, "retrain when\nperformance drops", color="#C0392B", fontsize=8, ha="center")
ax.set_title("Figure 1: End-to-end Machine Learning Workflow", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT + "w3_fig1_workflow.png", dpi=150); plt.close()

# Fig 2 - preprocessing pipeline
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 24); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 2.2, 5, 3.6, 2.2, "Raw features\n(X_train)", "#FFF2B3")
box(ax, 8.6, 8.2, 6.2, 2.2, "Numeric columns\nmedian impute → log(skewed)\n→ StandardScaler", "#CFE2F3")
box(ax, 8.6, 5.0, 6.2, 2.2, "Low-cardinality categorical\nmost-frequent impute\n→ OneHotEncoder", "#D9EAD3")
box(ax, 8.6, 1.8, 6.2, 2.2, "High-cardinality categorical\n(product category, state)\n→ Target / frequency encoding", "#D9D2E9")
for y in (8.2, 5.0, 1.8):
    arr(ax, (4.0, 5), (5.5, y))
    arr(ax, (11.7, y), (13.6, 5))
box(ax, 15.3, 5, 3.4, 2.4, "ColumnTransformer\n(combine)", "#FCE5CD")
box(ax, 20.4, 5, 3.8, 2.4, "Model\n(e.g. XGBoost,\nclass weights)", "#F4CCCC")
arr(ax, (17.0, 5), (18.5, 5))
ax.add_patch(Rectangle((4.8, 0.3), 17.7, 9.2, fill=False, ls="--", ec="grey"))
ax.text(13.6, 9.7, "sklearn Pipeline – fit only on training folds (prevents data leakage)",
        ha="center", fontsize=9, color="grey")
ax.set_title("Figure 2: Preprocessing and Model Pipeline", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout(); fig.savefig(OUT + "w3_fig2_pipeline.png", dpi=150); plt.close()

# Fig 3 - time-based split and CV
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 22); ax.set_ylim(-0.6, 7.6); ax.axis("off")
ax.text(0.2, 7.1, "(a) Overall split by purchase date", fontsize=10, fontweight="bold")
segs = [(1, 13, "#4C72B0", "Train  (~70%, oldest orders)"), (14, 3, "#DD8452", "Validation (~15%)"),
        (17, 3, "#C44E52", "Test (~15%, newest)")]
for x, w, c, t in segs:
    ax.add_patch(Rectangle((x, 5.6), w, 1.0, fc=c, ec="white"))
    ax.text(x + w / 2, 6.1, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
ax.annotate("", xy=(20.6, 5.3), xytext=(1, 5.3), arrowprops=dict(arrowstyle="->", color="grey"))
ax.text(10.8, 4.85, "time  →", ha="center", fontsize=8, color="grey")
ax.text(0.2, 4.1, "(b) TimeSeriesSplit inside the training period (expanding window)", fontsize=10, fontweight="bold")
for i in range(4):
    y = 3.1 - i * 0.85
    tr = 4 + i * 2.2
    ax.add_patch(Rectangle((1, y), tr, 0.6, fc="#4C72B0", ec="white"))
    ax.add_patch(Rectangle((1 + tr, y), 2.2, 0.6, fc="#55A868", ec="white"))
    ax.add_patch(Rectangle((1 + tr + 2.2, y), 13 - tr - 2.2, 0.6, fc="#EEEEEE", ec="white"))
    ax.text(0.6, y + 0.3, f"Fold {i + 1}", ha="right", va="center", fontsize=8)
ax.add_patch(Rectangle((15, 3.1), 0.6, 0.6, fc="#4C72B0")); ax.text(15.8, 3.4, "train", va="center", fontsize=8)
ax.add_patch(Rectangle((15, 2.3), 0.6, 0.6, fc="#55A868")); ax.text(15.8, 2.6, "validate", va="center", fontsize=8)
ax.add_patch(Rectangle((15, 1.5), 0.6, 0.6, fc="#EEEEEE")); ax.text(15.8, 1.8, "not used in this fold", va="center", fontsize=8)
ax.set_title("Figure 3: Time-based Split and Cross-Validation", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT + "w3_fig3_validation.png", dpi=150); plt.close()

# Fig 4 - confusion matrix explained
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.set_xlim(0, 10); ax.set_ylim(-0.6, 8); ax.axis("off")
cells = [(3, 4, "#D9EAD3", "True Negative\non-time order\ncorrectly predicted"),
         (6.5, 4, "#FCE5CD", "False Positive\non-time order\nwrongly flagged\n(small cost)"),
         (3, 1, "#F4CCCC", "False Negative\nlate order\nMISSED\n(big cost)"),
         (6.5, 1, "#CFE2F3", "True Positive\nlate order\ncaught early")]
for x, y, c, t in cells:
    ax.add_patch(Rectangle((x - 1.6, y - 1.3), 3.3, 2.8, fc=c, ec="#333"))
    ax.text(x + 0.05, y + 0.1, t, ha="center", va="center", fontsize=9)
ax.text(3, 6.1, "Predicted: on time", ha="center", fontsize=9, fontweight="bold")
ax.text(6.5, 6.1, "Predicted: late", ha="center", fontsize=9, fontweight="bold")
ax.text(1.0, 4.1, "Actual:\non time", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(1.0, 1.1, "Actual:\nlate", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(5, 7.3, "Precision = TP / (TP + FP)      Recall = TP / (TP + FN)", ha="center", fontsize=9.5)
ax.set_title("Figure 4: Confusion Matrix for this Problem", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT + "w3_fig4_confusion.png", dpi=150); plt.close()

# Fig 5 - deployment and maintenance loop
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 22); ax.set_ylim(0, 11); ax.axis("off")
nodes = [(4, 8.6, "New order\n(order system)", "#FFF2B3"),
         (11, 8.6, "Prediction API\n(FastAPI + saved\npipeline .pkl)", "#CFE2F3"),
         (18, 8.6, "Late-risk score\n→ support / logistics\ndashboard", "#D9EAD3"),
         (18, 2.4, "Log predictions +\nactual delivery\noutcome", "#FCE5CD"),
         (11, 2.4, "Monitoring\n(data drift, recall,\nweekly report)", "#F4CCCC"),
         (4, 2.4, "Retrain on recent\ndata (monthly or\nwhen recall drops)", "#D9D2E9")]
for x, y, t, c in nodes:
    box(ax, x, y, 4.6, 2.6, t, c)
arr(ax, (6.3, 8.6), (8.7, 8.6)); arr(ax, (13.3, 8.6), (15.7, 8.6)); arr(ax, (18, 7.3), (18, 3.7))
arr(ax, (15.7, 2.4), (13.3, 2.4)); arr(ax, (8.7, 2.4), (6.3, 2.4)); arr(ax, (4, 3.7), (11, 7.3), rad=-0.2)
ax.text(8.2, 4.4, "validate new model\nvs current, then\nreplace (versioned)", fontsize=8, color="#555")
ax.set_title("Figure 5: Deployment and Maintenance Cycle (optional stage)", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT + "w3_fig5_deployment.png", dpi=150); plt.close()
print("week 3 diagrams saved")
