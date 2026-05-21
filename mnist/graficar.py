import pickle
import numpy as np
import matplotlib.pyplot as plt

DIGIT_A    = 4
DIGIT_B    = 9
MODEL_NAME = "SimpleCNN_MNIST_4vs9_4capas"  # carpeta dentro de results/MNIST/

RESULTS_PATH = f"results/MNIST/{MODEL_NAME}/resultados/results.pkl"
OUTPUT_DIR   = f"results/MNIST/{MODEL_NAME}"

SUBGROUP_LABELS = {
    (0, 0): f"Sin patch, dígito {DIGIT_A}",
    (0, 1): f"Sin patch, dígito {DIGIT_B}",
    (1, 0): f"Con patch, dígito {DIGIT_A}",
    (1, 1): f"Con patch, dígito {DIGIT_B}",
}

with open(RESULTS_PATH, "rb") as f:
    raw = pickle.load(f)

# Agrupar por p1: {p1: {subgrupo: [acc, acc, ...]}}
grouped = {}
for entry in raw:
    p1 = entry["p1"]
    if p1 not in grouped:
        grouped[p1] = {key: [] for key in SUBGROUP_LABELS}
    for key in SUBGROUP_LABELS:
        grouped[p1][key].append(entry["subgroup_results"][key]["acc"])

P1_MAX    = 1  # graficá solo hasta este valor de p1
P1_VALUES = sorted(p for p in grouped.keys() if p <= P1_MAX)


def graficar_accuracies():
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, label in SUBGROUP_LABELS.items():
        means = [np.mean(grouped[p1][key]) for p1 in P1_VALUES]
        stds  = [np.std(grouped[p1][key])  for p1 in P1_VALUES]
        ax.plot(P1_VALUES, means, marker="o", label=label)
        ax.fill_between(P1_VALUES,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2)

    avg_means = [np.mean([np.mean(grouped[p1][key]) for key in SUBGROUP_LABELS]) for p1 in P1_VALUES]
    avg_stds  = [np.mean([np.std(grouped[p1][key])  for key in SUBGROUP_LABELS]) for p1 in P1_VALUES]
    ax.plot(P1_VALUES, avg_means, marker="o", linestyle="--", color="black", label="Promedio")
    ax.fill_between(P1_VALUES,
                    [m - s for m, s in zip(avg_means, avg_stds)],
                    [m + s for m, s in zip(avg_means, avg_stds)],
                    alpha=0.1, color="black")

    ax.set_xticks(P1_VALUES)
    ax.set_xticklabels([f"{p:.2f}" for p in P1_VALUES])
    ax.set_xlabel("p1 (proporción de dígito " + str(DIGIT_A) + " con patch en train)")
    ax.set_ylabel("Accuracy en test (balanceado)")
    ax.set_title(f"Accuracy por subgrupo vs. p1\n(media ± std, {len(raw) // len(P1_VALUES)} corridas)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/accuracy_{MODEL_NAME}.png"
    plt.savefig(out, dpi=150)
    print(f"Gráfico guardado en {out}")


def graficar_gap():
    grupo_A = [(0, 1), (1, 0)]  # con patch-dígito_a / sin patch-dígito_b
    grupo_B = [(0, 0), (1, 1)]  # sin patch-dígito_a / con patch-dígito_b

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

        accs_ben  = np.concatenate([grouped[p1][k] for k in beneficiados])
        accs_perj = np.concatenate([grouped[p1][k] for k in perjudicados])
        gaps.append(np.mean(accs_ben) - np.mean(accs_perj))
        gaps_std.append(np.sqrt(np.var(accs_ben) / len(accs_ben) + np.var(accs_perj) / len(accs_perj)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(P1_VALUES, gaps, marker="o", color="crimson")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="gray",  linestyle=":",  linewidth=0.8)
    ax.set_xticks(P1_VALUES)
    ax.set_xticklabels([f"{p:.2f}" for p in P1_VALUES])
    ax.set_xlabel("p1 (proporción de dígito " + str(DIGIT_A) + " con patch en train)")
    ax.set_ylabel("Gap de accuracy (beneficiados − perjudicados)")
    ax.set_title(f"Gap de accuracy entre grupos beneficiados y perjudicados vs. p1")
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/gap_{MODEL_NAME}.png"
    plt.savefig(out, dpi=150)
    print(f"Gráfico guardado en {out}")


graficar_accuracies()
graficar_gap()
plt.show()
