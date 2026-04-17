import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────
with open("benchmark_results.json") as f:
    raw = json.load(f)

EXPERIMENTS = [
    "dj_constant_3q",
    "dj_balanced_3q",
    "dj_balanced_4q",
    "grover_2q",
    "grover_3q",
    "grover_4q",
]
LABELS = {
    "dj_constant_3q": "DJ Const 3q",
    "dj_balanced_3q": "DJ Bal 3q",
    "dj_balanced_4q": "DJ Bal 4q",
    "grover_2q":      "Grover 2q",
    "grover_3q":      "Grover 3q",
    "grover_4q":      "Grover 4q",
}
BACKENDS = ["ideal", "noisy", "real"]
BACKEND_NAMES = {
    "ideal": "Ideal Simulator",
    "noisy": "Noisy Simulator",
    "real":  "Real Hardware",
}
COLORS = {
    "ideal": "#2196F3",
    "noisy": "#FF9800",
    "real":  "#F44336",
}

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for exp in EXPERIMENTS:
    for backend in BACKENDS:
        key = f"{exp}_{backend}"
        if key not in raw:
            continue
        d = raw[key]
        rows.append({
            "experiment":    exp,
            "label":         LABELS[exp],
            "backend":       backend,
            "success_prob":  d["success_probability"],
            "circuit_depth": d["circuit_depth"],
            "gate_count":    d["gate_count"],
            "counts":        d["counts"],
        })

df = pd.DataFrame(rows)

# ── Print summary tables ──────────────────────────────────────────────────────
print("\nSUCCESS PROBABILITY")
print("-" * 50)
pivot = df.pivot_table(index="experiment", columns="backend",
                       values="success_prob").reindex(EXPERIMENTS)[BACKENDS]
pivot.index = [LABELS[e] for e in EXPERIMENTS]
pivot.columns = ["Ideal", "Noisy Sim", "Real HW"]
print(pivot.map(lambda x: f"{x*100:.1f}%").to_string())

print("\nCIRCUIT DEPTH")
print("-" * 50)
depth = df.pivot_table(index="experiment", columns="backend",
                       values="circuit_depth").reindex(EXPERIMENTS)[BACKENDS]
depth.index = [LABELS[e] for e in EXPERIMENTS]
depth.columns = ["Ideal", "Noisy Sim", "Real HW"]
print(depth.to_string())

# ── Figure 1 — Success Probability Bar Chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(EXPERIMENTS))
width = 0.25

for i, backend in enumerate(BACKENDS):
    sub  = df[df.backend == backend].set_index("experiment").reindex(EXPERIMENTS)
    vals = sub["success_prob"].values * 100
    bars = ax.bar(x + (i - 1) * width, vals, width=width,
                  color=COLORS[backend], label=BACKEND_NAMES[backend])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS])
ax.set_ylabel("Success Probability (%)")
ax.set_title("Success Probability by Backend")
ax.set_ylim(0, 115)
ax.legend()
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig1_success_probability.png", dpi=150)
plt.close()
print("\nSaved: fig1_success_probability.png")

# ── Figure 2 — Circuit Depth ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for backend in BACKENDS:
    sub  = df[df.backend == backend].set_index("experiment").reindex(EXPERIMENTS)
    vals = sub["circuit_depth"].values
    ax.plot([LABELS[e] for e in EXPERIMENTS], vals,
            marker="o", label=BACKEND_NAMES[backend],
            color=COLORS[backend], linewidth=2)
    for xi, val in enumerate(vals):
        ax.annotate(str(int(val)), (xi, val),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=COLORS[backend])

ax.set_ylabel("Circuit Depth")
ax.set_title("Circuit Depth After Transpilation")
ax.legend()
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig2_circuit_depth.png", dpi=150)
plt.close()
print("Saved: fig2_circuit_depth.png")

# ── Figure 3 — Depth vs Success Scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for backend in ["noisy", "real"]:
    sub = df[df.backend == backend]
    ax.scatter(sub["circuit_depth"], sub["success_prob"] * 100,
               color=COLORS[backend], label=BACKEND_NAMES[backend],
               s=80, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row["label"],
                    (row["circuit_depth"], row["success_prob"] * 100),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)

ax.axhline(25, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.text(700, 27, "Random guess (4q)", fontsize=7, color="gray")
ax.set_xlabel("Circuit Depth (after transpilation)")
ax.set_ylabel("Success Probability (%)")
ax.set_title("Circuit Depth vs Success Probability")
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig3_depth_vs_success.png", dpi=150)
plt.close()
print("Saved: fig3_depth_vs_success.png")

# ── Figure 4 — Count Distributions for Grover 3Q ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

target   = "101"
all_keys = sorted(set(
    list(raw["grover_3q_ideal"]["counts"].keys()) +
    list(raw["grover_3q_noisy"]["counts"].keys()) +
    list(raw["grover_3q_real"]["counts"].keys())
))
all_keys = [target] + [k for k in all_keys if k != target]

for ax, backend in zip(axes, BACKENDS):
    counts = raw[f"grover_3q_{backend}"]["counts"]
    vals   = [counts.get(k, 0) for k in all_keys]
    colors = [COLORS[backend] if k == target else "lightgray" for k in all_keys]

    ax.barh(all_keys, vals, color=colors)
    ax.set_title(BACKEND_NAMES[backend])
    ax.set_xlabel("Shot count")
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    prob = raw[f"grover_3q_{backend}"]["success_probability"]
    ax.text(0.97, 0.03, f"Success: {prob*100:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[0].set_ylabel("Bitstring")
fig.suptitle("Grover 3Q — Measurement Distribution (Target: '101')")

plt.tight_layout()
plt.savefig("fig4_count_distributions.png", dpi=150)
plt.close()
print("Saved: fig4_count_distributions.png")

# ── Figure 5 — Heatmap ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

heat = df.pivot_table(index="label", columns="backend",
                      values="success_prob").reindex(
    [LABELS[e] for e in EXPERIMENTS])[BACKENDS]

im = ax.imshow(heat.values, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")

ax.set_xticks(range(len(BACKENDS)))
ax.set_xticklabels(["Ideal", "Noisy Sim", "Real HW"])
ax.set_yticks(range(len(EXPERIMENTS)))
ax.set_yticklabels(heat.index)

for i in range(len(EXPERIMENTS)):
    for j in range(len(BACKENDS)):
        val = heat.values[i, j]
        ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black" if val > 0.55 else "white")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Success Probability")
ax.set_title("Success Probability Heatmap")

plt.tight_layout()
plt.savefig("fig5_heatmap.png", dpi=150)
plt.close()
print("Saved: fig5_heatmap.png")