"""Generates the planning diagrams used in the Week 1 report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)

def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", rad=0.0, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=1.3, ls=ls, color=color, connectionstyle=f"arc3,rad={rad}"))

# 1. Project workflow (CRISP-DM based)
fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")
steps = [("1. Business Understanding\nDefine 'late delivery', success criteria", "#FDE2C8"),
         ("2. Data Collection\nOlist e-commerce CSV files (9 tables)", "#FFF2B3"),
         ("3. Data Cleaning & Merging\nOne row per delivered order", "#D9EAD3"),
         ("4. Exploratory Data Analysis\nPatterns by region, month, category", "#CFE2F3"),
         ("5. Feature Engineering\nDistance, estimate window, freight ratio", "#D9D2E9"),
         ("6. Modelling\nLogistic Regression → Random Forest → XGBoost", "#F4CCCC"),
         ("7. Evaluation\nRecall, F1, ROC-AUC, confusion matrix", "#FCE5CD"),
         ("8. Reporting & Presentation\nFindings, recommendations, GitHub", "#D0E0E3")]
ys = [12.2 - i*1.55 for i in range(len(steps))]
for (t, c), y in zip(steps, ys):
    box(ax, 5, y, 6.4, 1.05, t, c, 9.5)
for y1, y2 in zip(ys[:-1], ys[1:]):
    arrow(ax, 5, y1 - 0.53, 5, y2 + 0.53)
arrow(ax, 8.25, ys[6], 8.25, ys[2], rad=0.35, ls="--", color="#C0392B")
ax.text(9.55, (ys[6] + ys[2]) / 2, "Go back if results\nare not good enough", fontsize=8,
        color="#C0392B", rotation=90, ha="center", va="center")
ax.set_title("Figure 1: Project Workflow (based on CRISP-DM)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig("images/fig1_workflow.png", dpi=160); plt.close()

# 2. Data pipeline / architecture
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis("off")
tables = ["orders", "order_items", "customers", "sellers", "products", "geolocation"]
for i, t in enumerate(tables):
    y = 9 - i * 1.5
    box(ax, 1.8, y, 3.0, 0.95, t + ".csv", "#FFF2B3", 9)
    arrow(ax, 3.3, y, 5.2, 5.2)
pipe = [(6.9, "Merge on\norder_id /\ncustomer_id /\nseller_id", "#D9EAD3"),
        (10.1, "Clean\n(datetime,\nmissing values,\ndelivered only)", "#CFE2F3"),
        (13.3, "Feature\nEngineering\n+ target is_late", "#D9D2E9"),
        (16.5, "Train / Test\nsplit by date\n+ models", "#F4CCCC")]
for x, t, c in pipe:
    box(ax, x, 5.2, 2.7, 2.6, t, c, 9)
for (x1, _, _), (x2, _, _) in zip(pipe[:-1], pipe[1:]):
    arrow(ax, x1 + 1.35, 5.2, x2 - 1.35, 5.2)
box(ax, 20.2, 7.2, 2.9, 1.5, "Evaluation\nmetrics +\nfeature importance", "#FCE5CD", 8.5)
box(ax, 20.2, 3.2, 2.9, 1.5, "Report, notebook\n& optional\nStreamlit demo", "#D0E0E3", 8.5)
arrow(ax, 17.85, 5.6, 18.75, 7.0); arrow(ax, 17.85, 4.8, 18.75, 3.4)
box(ax, 10.6, 0.9, 9.6, 1.0, "order_reviews → used only in EDA (not as a feature, to avoid leakage)", "#EEEEEE", 8.5)
ax.set_title("Figure 2: Proposed Data Pipeline", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig("images/fig2_pipeline.png", dpi=160); plt.close()

# 3. Timeline (Gantt chart, ~32 hours)
tasks = [("Problem understanding & research", 0, 3, "Week 1"),
         ("Dataset study & project plan", 3, 3, "Week 1"),
         ("Data cleaning & merging", 6, 5, "Week 2"),
         ("Exploratory data analysis", 11, 5, "Week 2"),
         ("Feature engineering", 16, 4, "Week 3"),
         ("Model building & tuning", 20, 6, "Week 3"),
         ("Evaluation & interpretation", 26, 3, "Week 4"),
         ("Final report & GitHub upload", 29, 3, "Week 4")]
colors = {"Week 1": "#F39C12", "Week 2": "#27AE60", "Week 3": "#2980B9", "Week 4": "#8E44AD"}
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (name, start, dur, wk) in enumerate(tasks):
    ax.barh(i, dur, left=start, color=colors[wk], edgecolor="black", height=0.55)
    ax.text(start + dur / 2, i, f"{dur} h", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks]); ax.invert_yaxis()
ax.set_xlabel("Cumulative hours of work"); ax.set_xlim(0, 33); ax.set_xticks(range(0, 34, 4))
ax.grid(axis="x", ls=":", alpha=0.6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=c, ec="black", label=wk) for wk, c in colors.items()],
          loc="upper right", fontsize=8)
ax.set_title("Figure 3: Project Timeline (total ≈ 32 hours)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig("images/fig3_timeline.png", dpi=160); plt.close()
print("diagrams saved in images/")
