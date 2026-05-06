import pickle
import numpy as np
import matplotlib.pyplot as plt

# Cambiá estos valores según el experimento
BIAS_ATTR      = "Young"
BIAS_LABEL0    = "jóvenes"
BIAS_LABEL1    = "adultos"
TARGET_LABEL_M = "rubios"
TARGET_LABEL_F = "rubios"

SUBGROUP_LABELS = {
    (0, 0): f"{BIAS_LABEL0.capitalize()} no {TARGET_LABEL_F}",
    (0, 1): f"{BIAS_LABEL0.capitalize()} {TARGET_LABEL_F}",
    (1, 0): f"{BIAS_LABEL1.capitalize()} no {TARGET_LABEL_M}",
    (1, 1): f"{BIAS_LABEL1.capitalize()} {TARGET_LABEL_M}",
}

PICKLES = [
    f"results/results_{BIAS_ATTR}_cruzado.pkl"
]

# Cargar y concatenar resultados
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


def graficar_accuracies():
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
    ax.set_xlabel(f"Proporción de {BIAS_LABEL1} no {TARGET_LABEL_M} en train")
    ax.set_ylabel("Accuracy en test (balanceado)")
    ax.set_title(f"Accuracy por subgrupo vs. proporción de {BIAS_LABEL1} no {TARGET_LABEL_M} en train\n(media ± std, 10 corridas)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"results/accuracy_{BIAS_ATTR}_cruzado.png", dpi=150)
    print(f"Gráfico guardado en results/accuracy_{BIAS_ATTR}_cruzado.png")


def graficar_gap():
    # Con p2=1-p1, los grupos (0,1) y (1,0) tienen p1*n muestras
    # y (0,0) y (1,1) tienen (1-p1)*n muestras.
    # Beneficiados = los que tienen más muestras en train.
    grupo_A = [(0, 1), (1, 0)]  # cuentan p1*n
    grupo_B = [(0, 0), (1, 1)]  # cuentan (1-p1)*n

    gaps, gaps_std = [], []
    for p1 in P1_VALUES:
        if p1 > 0.5:
            beneficiados, perjudicados = grupo_A, grupo_B
        elif p1 < 0.5:
            beneficiados, perjudicados = grupo_B, grupo_A
        else:
            gaps.append(0.0)
            gaps_std.append(0.0)
            continue

        accs_ben  = np.concatenate([results_combined[p1][k] for k in beneficiados])
        accs_perj = np.concatenate([results_combined[p1][k] for k in perjudicados])
        gaps.append(np.mean(accs_ben) - np.mean(accs_perj))
        gaps_std.append(np.sqrt(np.var(accs_ben)/len(accs_ben) + np.var(accs_perj)/len(accs_perj)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(P1_VALUES, gaps, marker="o", color="crimson")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(P1_VALUES)
    ax.set_xticklabels([f"{p:.1f}" for p in P1_VALUES])
    ax.set_xlabel(f"Proporción de {BIAS_LABEL1} no {TARGET_LABEL_M} en train")
    ax.set_ylabel("Gap de accuracy (beneficiados − perjudicados)")
    ax.set_title(f"Gap de accuracy entre grupos beneficiados y perjudicados\nvs. proporción de {BIAS_LABEL1} no {TARGET_LABEL_M} en train")
    ax.set_ylim(-0.1, 0.6)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/gap_{BIAS_ATTR}_cruzado.png", dpi=150)
    print(f"Gráfico guardado en results/gap_{BIAS_ATTR}_cruzado.png")


graficar_accuracies()
graficar_gap()
plt.show()
