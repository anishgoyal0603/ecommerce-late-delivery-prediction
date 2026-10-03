"""
Mock-up visualizations for the final report and presentation plan (Week 4).

IMPORTANT: this is a hypothetical project. All numbers below are
ILLUSTRATIVE MOCK-UP VALUES chosen to show how the final charts would look
and how the story would be told. They are not results from real data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "images/"
LATE, ONTIME, GREY, ACCENT = "#C44E52", "#55A868", "#B0B0B0", "#4C72B0"
TAG = "Mock-up – illustrative values"


def tag(fig):
    fig.text(0.99, 0.01, TAG, ha="right", fontsize=7, color="grey", style="italic")


# 1. KPI summary tiles
fig, ax = plt.subplots(figsize=(11, 2.3)); ax.axis("off")
kpis = [("~8%", "of orders delivered late"), ("1.8 ★", "lower avg review\nwhen late (2.6 vs 4.4)"),
        ("62%", "of late orders flagged\nin advance by the model"), ("3 of 10", "flagged orders are\ntruly late (precision)")]
for i, (v, t) in enumerate(kpis):
    x = 0.02 + i * 0.245
    ax.add_patch(FancyBboxPatch((x, 0.08), 0.22, 0.84, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#F5F7FA", ec="#D0D7E2", transform=ax.transAxes))
    ax.text(x + 0.11, 0.62, v, ha="center", fontsize=22, fontweight="bold",
            color=LATE if i < 2 else ACCENT, transform=ax.transAxes)
    ax.text(x + 0.11, 0.25, t, ha="center", fontsize=9, color="#333", transform=ax.transAxes)
ax.set_title("Figure 1: Key numbers at a glance", fontweight="bold", loc="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig1_kpis.png", dpi=150); plt.close()

# 2. Trend with anomalies
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
late = [6.1, 13.5, 15.2, 7.4, 6.0, 5.2, 5.6, 6.3, 5.0, 5.8, 13.9, 9.7]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(months, late, marker="o", color=ACCENT, lw=2)
ax.axhline(np.mean(late), ls="--", color=GREY); ax.text(11.2, np.mean(late) + .3, "average", color="grey", fontsize=8)
for i in (1, 2, 10):
    ax.scatter(months[i], late[i], s=160, facecolors="none", edgecolors=LATE, lw=2)
ax.annotate("Black Friday / holiday\npeak", xy=(10, 13.9), xytext=(7.2, 13.6), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate("Post-holiday backlog +\ncarnival / strikes", xy=(2, 15.2), xytext=(3.4, 14.4), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_ylabel("% orders late"); ax.set_ylim(0, 18)
ax.set_title("Figure 3: Late deliveries spike in three months (trend + anomalies)", fontweight="bold", loc="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig2_trend.png", dpi=150); plt.close()

# 3. Region bar (sorted, highlight)
regions = ["North", "Northeast", "Centre-West", "Southeast", "South"]
rate = [14.8, 12.9, 8.6, 7.1, 6.4]
fig, ax = plt.subplots(figsize=(8, 3.4))
bars = ax.barh(regions[::-1], rate[::-1], color=[GREY, GREY, GREY, LATE, LATE])
for b, v in zip(bars, rate[::-1]):
    ax.text(v + .2, b.get_y() + b.get_height() / 2, f"{v}%", va="center", fontsize=9)
ax.set_xlabel("% orders late"); ax.set_xlim(0, 17)
ax.set_title("Figure 4: Remote regions have about twice the late rate", fontweight="bold", loc="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig3_region.png", dpi=150); plt.close()

# 4. Heatmap distance x estimate
d = ["< 300 km", "300–700", "700–1,500", "> 1,500 km"]
e = ["< 15 days", "15–25", "25–35", "> 35 days"]
vals = np.array([[5, 3, 2, 1], [10, 6, 4, 2], [19, 11, 6, 3], [31, 18, 10, 5]])
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(vals, cmap="Reds")
ax.set_xticks(range(4)); ax.set_xticklabels(e); ax.set_yticks(range(4)); ax.set_yticklabels(d)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{vals[i, j]}%", ha="center", va="center", color="white" if vals[i, j] > 15 else "black")
ax.set_xlabel("Delivery time promised to customer"); ax.set_ylabel("Seller–customer distance")
ax.set_title("Figure 5: Long distance + tight promise = highest risk", fontweight="bold", loc="left")
for s in ax.spines.values(): s.set_visible(False)
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig4_heatmap.png", dpi=150); plt.close()

# 5. Review impact
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(["On time", "Late"], [4.4, 2.6], color=[ONTIME, LATE], width=.55)
for i, v in enumerate([4.4, 2.6]): ax[0].text(i, v + .08, f"{v} ★", ha="center", fontweight="bold")
ax[0].set_ylim(0, 5.2); ax[0].set_title("Average review score")
stars = np.arange(1, 6)
on = [6, 3, 8, 20, 63]; lt = [44, 11, 14, 13, 18]
ax[1].bar(stars - .2, on, .4, color=ONTIME, label="On time"); ax[1].bar(stars + .2, lt, .4, color=LATE, label="Late")
ax[1].set_xticks(stars); ax[1].set_xticklabels([f"{s}★" for s in stars]); ax[1].set_ylabel("% of reviews"); ax[1].legend(frameon=False)
ax[1].set_title("Share of each rating")
fig.suptitle("Figure 2: Late delivery is the clearest driver of 1-star reviews", fontweight="bold", x=0.02, ha="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig5_reviews.png", dpi=150); plt.close()

# 6. Model comparison + feature importance
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
models = ["Dummy", "Logistic\nRegression", "Decision\nTree", "Random\nForest", "XGBoost"]
pr = [0.08, 0.27, 0.24, 0.36, 0.41]
ax[0].bar(models, pr, color=[GREY, GREY, GREY, GREY, ACCENT])
for i, v in enumerate(pr): ax[0].text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9)
ax[0].set_ylabel("PR-AUC (higher is better)"); ax[0].set_ylim(0, .5); ax[0].set_title("Model comparison")
feats = ["Distance (km)", "Days promised", "Peak season", "Customer region", "Seller past late %",
         "Freight / price", "Product weight", "Approval delay"]
imp = [0.24, 0.19, 0.14, 0.12, 0.11, 0.08, 0.07, 0.05]
ax[1].barh(feats[::-1], imp[::-1], color=ACCENT); ax[1].set_xlabel("relative importance (SHAP)")
ax[1].set_title("What drives the risk")
fig.suptitle("Figure 6: Gradient boosting performs best; distance and promised days matter most",
             fontweight="bold", x=0.02, ha="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig6_model.png", dpi=150); plt.close()

# 7. Threshold trade-off in business terms
thr = np.linspace(0.1, 0.8, 50)
recall = 0.95 - 0.95 * (thr - 0.1) ** 0.8
precision = 0.12 + 0.69 * (thr - 0.1) ** 0.9
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(thr, recall * 100, color=LATE, lw=2, label="% of late orders caught (recall)")
ax.plot(thr, precision * 100, color=ACCENT, lw=2, label="% of alerts that are real (precision)")
t0 = 0.366
r0 = 0.95 - 0.95 * (t0 - 0.1) ** 0.8; p0 = 0.12 + 0.69 * (t0 - 0.1) ** 0.9
ax.axvline(t0, color="black", ls="--")
ax.scatter([t0, t0], [r0 * 100, p0 * 100], color="black", zorder=5)
ax.text(t0 + 0.01, 88, f"chosen setting\n{r0:.0%} caught, {p0:.0%} real", fontsize=9)
ax.set_xlabel("Model alert threshold"); ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend(frameon=False, loc="upper right")
ax.set_title("Figure 7: Trade-off between catching late orders and false alarms", fontweight="bold", loc="left")
tag(fig); plt.tight_layout(); fig.savefig(OUT + "w4_fig7_tradeoff.png", dpi=150); plt.close()

# 8. Presentation storyboard
fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 24); ax.set_ylim(0, 8.4); ax.axis("off")
slides = [("1", "Hook", "1 in 12 orders\narrives late"), ("2", "Why it\nmatters", "Late = 1.8★\nlower rating"),
          ("3", "Where &\nwhen", "Remote regions,\nNov & Feb–Mar"), ("4", "Root\ncause", "Distance + tight\npromise"),
          ("5", "Solution", "Model flags 62%\nin advance"), ("6", "Trade-off", "3 in 10 alerts\nare real"),
          ("7", "Actions", "3 recommen-\ndations"), ("8", "Next\nsteps", "Pilot + ask")]
phase = ["Situation"] * 2 + ["Complication"] * 2 + ["Resolution"] * 4
pc = {"Situation": "#FFF2B3", "Complication": "#F4CCCC", "Resolution": "#D9EAD3"}
for i, ((n, t, s), p) in enumerate(zip(slides, phase)):
    x = 1.5 + i * 2.95
    ax.add_patch(FancyBboxPatch((x - 1.25, 2.0), 2.5, 4.2, boxstyle="round,pad=0.02,rounding_size=0.15", fc=pc[p], ec="#333"))
    ax.text(x, 5.6, f"Slide {n}", ha="center", fontsize=8, color="#555")
    ax.text(x, 4.6, t, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x, 3.0, s, ha="center", va="center", fontsize=8.5)
    if i < 7:
        ax.add_patch(FancyArrowPatch((x + 1.27, 4.1), (x + 1.68, 4.1), arrowstyle="-|>", mutation_scale=10))
for p, x0, x1 in [("Situation", 0.25, 5.7), ("Complication", 6.15, 11.6), ("Resolution", 12.05, 23.5)]:
    ax.plot([x0, x1], [1.4, 1.4], color="#333", lw=1); ax.text((x0 + x1) / 2, 0.8, p, ha="center", fontweight="bold")
ax.set_title("Figure 8: Presentation storyboard (Situation → Complication → Resolution)", fontweight="bold", loc="left")
plt.tight_layout(); fig.savefig(OUT + "w4_fig8_storyboard.png", dpi=150); plt.close()
print("Week 4 mock-up charts saved")
