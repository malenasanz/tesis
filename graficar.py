import pickle
import numpy as np
import matplotlib.pyplot as plt

SUBGROUP_LABELS = {
    (0, 0): "Mujer no rubia",
    (0, 1): "Mujer rubia",
    (1, 0): "Hombre no rubio",
    (1, 1): "Hombre rubio",
}

PICKLES = [
    "results/results_vs_p1.pkl",
    "results/results_vs_p1_2.pkl",
]

# Concatena las accuracies de todos los pickles por p1 y subgrupo
results_combined = {}
for path in PICKLES:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for p1 in data["p1_values"]:
        if p1 not in results_combined:
            results_combined[p1] = {key: [] for key in SUBGROUP_LABELS}
        for key in SUBGROUP_LABELS:
            results_combined[p1][key].extend(data["results"][p1][key])

P1_VALUES = sorted(results_combined.keys())

fig, ax = plt.subplots(figsize=(10, 6))
for key, label in SUBGROUP_LABELS.items():
    means = [np.mean(results_combined[p1][key]) for p1 in P1_VALUES]
    stds  = [np.std(results_combined[p1][key])  for p1 in P1_VALUES]
    ax.plot(P1_VALUES, means, marker="o", label=label)
    ax.fill_between(P1_VALUES,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2)

avg_means = [np.mean([np.mean(results_combined[p1][key]) for key in SUBGROUP_LABELS]) for p1 in P1_VALUES]
avg_stds  = [np.mean([np.std(results_combined[p1][key])  for key in SUBGROUP_LABELS]) for p1 in P1_VALUES]
ax.plot(P1_VALUES, avg_means, marker="o", linestyle="--", color="black", label="Promedio")
ax.fill_between(P1_VALUES,
                [m - s for m, s in zip(avg_means, avg_stds)],
                [m + s for m, s in zip(avg_means, avg_stds)],
                alpha=0.1, color="black")

ax.set_xticks(P1_VALUES)
ax.set_xticklabels([f"{p:.1f}" for p in P1_VALUES])
ax.set_xlabel("p1 (proporción de hombres no rubios)")
ax.set_ylabel("Accuracy en test")
ax.set_title("Accuracy por subgrupo vs p1 (p2 = 1 - p1)")
ax.legend()
ax.grid(True)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("results/accuracy_vs_p1_todas_proporciones_con_promedio.png", dpi=150)
plt.show()
print("Gráfico guardado en results/accuracy_vs_p1_todas_proporciones_con_promedio.png")
