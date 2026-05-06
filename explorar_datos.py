import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/raw/celeba/list_attr_celeba.txt", sep=r"\s+", header=1, index_col=0)
df = (df + 1) // 2

print(f"Total de imágenes: {len(df)}")
print(f"Total de atributos: {len(df.columns)}")
print(f"Hombres: {df['Male'].sum()} ({df['Male'].mean()*100:.1f}%)")
print(f"Mujeres: {(1-df['Male']).sum()} ({(1-df['Male']).mean()*100:.1f}%)")

ATTRS = {
    "Blond_Hair": "Pelo rubio",
    "Brown_Hair": "Pelo marrón",
    "Young":      "Joven",
    "Pale_Skin":  "Piel clara",
}

hombres = df[df["Male"] == 1]
mujeres = df[df["Male"] == 0]

counts_h = [hombres[a].sum() for a in ATTRS]
counts_m = [mujeres[a].sum() for a in ATTRS]

for a, label in ATTRS.items():
    ch, cm = hombres[a].sum(), mujeres[a].sum()
    print(f"{label} — Hombres: {ch} ({ch/len(hombres)*100:.1f}%) | Mujeres: {cm} ({cm/len(mujeres)*100:.1f}%)")

print("\n--- Tamaño de subgrupos (bias_attr x Blond_Hair) ---")
for bias_attr, b0, b1 in [("Male", "Mujer", "Hombre"), ("Young", "Mayor", "Joven")]:
    print(f"\n  bias_attr = {bias_attr}:")
    min_n = None
    for bv, blabel in [(0, b0), (1, b1)]:
        for tv, tlabel in [(0, "no rubia/o"), (1, "rubia/o")]:
            n = len(df[(df[bias_attr] == bv) & (df["Blond_Hair"] == tv)])
            print(f"    {blabel} {tlabel}: {n}")
            if min_n is None or n < min_n:
                min_n = n
    print(f"  → Subgrupo mínimo: {min_n}")

x = np.arange(len(ATTRS))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, counts_h, width, label="Hombres")
ax.bar(x + width/2, counts_m, width, label="Mujeres")

ax.set_xticks(x)
ax.set_xticklabels(list(ATTRS.values()))
ax.set_ylabel("Cantidad de positivos")
ax.set_title(f"Positivos por atributo y género (total: {len(df):,} imágenes)")
ax.legend()
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("results/histograma_atributos.png", dpi=150)
plt.show()
print("Gráfico guardado en results/histograma_atributos.png")
