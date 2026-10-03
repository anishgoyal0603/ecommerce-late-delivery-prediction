"""
Makes the example charts used in the Week 2 report.

NOTE: Week 2 is a planning task, so no real dataset is analysed yet.
The data below is a small SIMULATED sample with the same columns I expect
after cleaning the Olist data. It is only used to show what each type of
chart will look like. The real numbers will come in Week 3.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sns.set_theme(style="whitegrid", palette="Set2")
rng = np.random.default_rng(42)
OUT = "images/"
NOTE = "Illustrative example – simulated sample data"

# ---------- simulated sample ----------
n = 2000
distance = rng.gamma(2.0, 280, n)                       # km
weight = rng.lognormal(7, 1, n)                         # grams
price = rng.lognormal(4.5, 0.7, n)
freight = 8 + distance * 0.02 + weight / 1500 + rng.normal(0, 4, n)
est_days = np.clip(rng.normal(23, 8, n), 5, 60).round()
month = rng.integers(1, 13, n)
states = rng.choice(["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "PE", "CE", "MA"], n,
                    p=[.40, .13, .12, .06, .05, .05, .05, .05, .05, .04])
logit = -3.4 + distance / 900 - est_days / 40 + (month == 11) * 0.9 + (month == 3) * 0.6 \
        + np.isin(states, ["BA", "MA", "CE", "PE"]) * 0.8
late = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
review = np.where(late == 1, rng.choice([1, 2, 3, 4, 5], n, p=[.45, .15, .15, .12, .13]),
                  rng.choice([1, 2, 3, 4, 5], n, p=[.07, .03, .08, .2, .62]))
df = pd.DataFrame({"distance_km": distance, "product_weight_g": weight, "price": price,
                   "freight_value": freight, "estimated_days": est_days, "month": month,
                   "customer_state": states, "is_late": late, "review_score": review})
df.loc[rng.choice(n, 60, replace=False), "product_weight_g"] = np.nan


def note(fig):
    fig.text(0.99, 0.01, NOTE, ha="right", fontsize=7, color="grey", style="italic")


# 1. univariate - histogram + boxplot
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.histplot(df["distance_km"], kde=True, ax=ax[0], color="#4C72B0")
ax[0].set_title("Histogram: seller–customer distance")
sns.boxplot(x=df["freight_value"], ax=ax[1], color="#DD8452")
ax[1].set_title("Boxplot: freight value (outliers visible)")
note(fig); plt.tight_layout(); fig.savefig(OUT + "w2_fig2_univariate.png", dpi=150); plt.close()

# 2. categorical / target balance
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
counts = df["is_late"].map({0: "On time", 1: "Late"}).value_counts()
ax[0].bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
ax[0].set_title("Bar chart: target class balance")
ax[0].set_ylim(0, counts.max() * 1.15)
for i, v in enumerate(counts.values):
    ax[0].text(i, v, f"{v / n:.0%}", ha="center", va="bottom")
st = df["customer_state"].value_counts()
sns.barplot(x=st.values, y=st.index, ax=ax[1], color="#8172B3")
ax[1].set_title("Bar chart: orders by customer state")
note(fig); plt.tight_layout(); fig.savefig(OUT + "w2_fig3_categorical.png", dpi=150); plt.close()

# 3. bivariate
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
sns.scatterplot(data=df.sample(600, random_state=1), x="distance_km", y="freight_value",
                hue="is_late", alpha=.6, ax=ax[0], palette={0: "#55A868", 1: "#C44E52"})
ax[0].set_title("Scatter: distance vs freight")
sns.boxplot(data=df, x="is_late", y="distance_km", ax=ax[1], palette=["#55A868", "#C44E52"],
            hue="is_late", legend=False)
ax[1].set_title("Boxplot: distance by late / on time")
rate = df.groupby("month")["is_late"].mean() * 100
ax[2].plot(rate.index, rate.values, marker="o", color="#C44E52")
ax[2].set_xticks(range(1, 13)); ax[2].set_title("Line: late % by purchase month")
ax[2].set_ylabel("% late")
note(fig); plt.tight_layout(); fig.savefig(OUT + "w2_fig4_bivariate.png", dpi=150); plt.close()

# 4. multivariate - heatmap + pivot heatmap
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
num = df[["distance_km", "product_weight_g", "price", "freight_value", "estimated_days",
          "review_score", "is_late"]]
corr = num.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0],
            mask=np.triu(np.ones_like(corr, dtype=bool)), cbar=False, annot_kws={"size": 8})
ax[0].set_title("Correlation heatmap")
df["dist_band"] = pd.cut(df["distance_km"], [0, 300, 700, 1500, 10000],
                         labels=["<300", "300-700", "700-1500", ">1500"])
df["est_band"] = pd.cut(df["estimated_days"], [0, 15, 25, 35, 100],
                        labels=["<15", "15-25", "25-35", ">35"])
pv = df.pivot_table(index="dist_band", columns="est_band", values="is_late",
                    aggfunc="mean", observed=False) * 100
sns.heatmap(pv, annot=True, fmt=".0f", cmap="Reds", ax=ax[1])
ax[1].set_title("Late % by distance band × estimate window")
ax[1].set_xlabel("estimated days"); ax[1].set_ylabel("distance (km)")
ax[1].tick_params(axis="y", rotation=0)
note(fig); plt.tight_layout(); fig.savefig(OUT + "w2_fig5_multivariate.png", dpi=150); plt.close()

# 5. missing values
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(df.drop(columns=["dist_band", "est_band"]).isna().T, cbar=False, cmap=["#EEEEEE", "#C44E52"], ax=ax,
            xticklabels=False)
ax.set_title("Missing value map (red = missing)")
note(fig); plt.tight_layout(); fig.savefig(OUT + "w2_fig6_missing.png", dpi=150); plt.close()


# 6. EDA framework flowchart
def box(ax, x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=c, ec="#333", lw=1.2))
    ax.text(x, y, t, ha="center", va="center", fontsize=9)


def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))


fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 24); ax.set_ylim(0, 10); ax.axis("off")
top = [(2.2, "1. Load &\nfirst look\n(shape, dtypes,\nhead)", "#FFF2B3"),
       (6.6, "2. Data quality\n(missing values,\nduplicates,\nwrong types)", "#D9EAD3"),
       (11.0, "3. Univariate\n(distributions,\ncounts, skew)", "#CFE2F3"),
       (15.4, "4. Bivariate\n(feature vs\ntarget)", "#D9D2E9"),
       (19.8, "5. Multivariate\n(correlation,\npivot heatmaps)", "#F4CCCC")]
for x, t, c in top:
    box(ax, x, 7, 3.6, 2.6, t, c)
for (x1, _, _), (x2, _, _) in zip(top[:-1], top[1:]):
    arr(ax, (x1 + 1.8, 7), (x2 - 1.8, 7))
bottom = [(19.8, "6. Outlier\ntreatment\n(IQR, capping)", "#FCE5CD"),
          (15.4, "7. Feature ideas\n& hypotheses\nlist", "#D0E0E3"),
          (11.0, "8. Document\n(notebook +\nreport)", "#EAD1DC")]
for x, t, c in bottom:
    box(ax, x, 2.6, 3.6, 2.4, t, c)
arr(ax, (19.8, 5.7), (19.8, 3.8))
arr(ax, (18.0, 2.6), (17.2, 2.6)); arr(ax, (13.6, 2.6), (12.8, 2.6))
ax.annotate("", xy=(6.6, 5.7), xytext=(9.2, 2.6),
            arrowprops=dict(arrowstyle="-|>", ls="--", color="#C0392B", connectionstyle="arc3,rad=-0.3"))
ax.text(5.2, 3.0, "new issue found?\ngo back and re-clean", color="#C0392B", fontsize=8, ha="center")
ax.set_title("Figure 1: EDA Framework – order of steps", fontsize=12, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT + "w2_fig1_framework.png", dpi=150); plt.close()
print("Week 2 charts saved in images/")
