#from src.datasets import CelebADataSet 
#from src.dataloader import _get_groups_balanced, _stratified_split 
import pandas as pd 

import pandas as pd

df = pd.read_csv(
    "data/raw/celeba/list_attr_celeba.txt",
    sep="\s+",
    skiprows=1
)
df = (df+1)//2

def split_por_subgrupos(df, bias_attr, target_attr, split_size=0.1, seed=42):
    grupos = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
              for b in [0, 1] for t in [0, 1]}
    n = min(len(g) for g in grupos.values())
    n_split = int(split_size * n)

    main_dfs, split_dfs = [], []
    for g in grupos.values():
        g = g.sample(frac=1, random_state=seed)
        split_dfs.append(g.iloc[:n_split])
        main_dfs.append(g.iloc[n_split:])

    main_df  = pd.concat(main_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    split_df = pd.concat(split_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    return main_df, split_df


def get_subset_con_proporcion(df, bias_attr, target_attr, p, N, seed=42):
    # Los 4 grupos
    grupos = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
         for b in [0, 1] for t in [0, 1]}
    min_n = min(len(g) for g in grupos.values())
    hombres_rubios = min(min_n, int(p * N), N // 2)
    hombres_no_rubios = int(p * N) - hombres_rubios

    counts = {
        (1, 1): hombres_rubios, #hombres_rubios
        (1, 0): hombres_no_rubios, #hombres_no_rubios
        (0, 1): N // 2 - hombres_rubios, #mujeres_rubias
        (0, 0): N // 2 - hombres_no_rubios, #mujers_no_rubias
    }

    for key, n in counts.items():
        if n < 0:
            raise ValueError(f"Grupo {key}: cantidad negativa ({n}). Revisá p o N.")
        if n > len(grupos[key]):
            raise ValueError(f"Grupo {key}: se piden {n} pero solo hay {len(grupos[key])} disponibles.")

    result = pd.concat([grupos[key].sample(n=n, random_state=seed) for key, n in counts.items()])
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)



#minimo_subgrupo(df,"Male","Blond_Hair")
#print("Dataset: ", len(df))
#print("80% dataset: "   , int(0.8*len(df)))
#print("10% dataset: "   , int(0.1*len(df)))
#print(print("10% dataset balanceado: "   , int(0.1*minimo_subgrupo(df,"Male","Blond_Hair"))))

bias_attr = "Male"
target_attr = "Blond_Hair"
desarrollo, test = split_por_subgrupos(df, bias_attr, target_attr, split_size=0.1)
train_balanceado, val_balanceado = split_por_subgrupos(desarrollo, bias_attr, target_attr, split_size=0.1)
N_max = len(train_balanceado)
desarrollo_proporcion = get_subset_con_proporcion(desarrollo, bias_attr, target_attr, p=0.5, N=N_max)
print("Cantidad datos desarrollo: ", len(desarrollo))
for (bias_value, target_value), group in desarrollo.groupby([bias_attr, target_attr]):
        print(f"Grupo ({bias_value}, {target_value}): {len(group)} muestras")
print("Cantidad datos test: ", len(test))
for (bias_value, target_value), group in test.groupby([bias_attr, target_attr]):
        print(f"Grupo ({bias_value}, {target_value}): {len(group)} muestras")
print("Cantidad datos desarrollo proporcional: ", len(desarrollo_proporcion))
for (bias_value, target_value), group in desarrollo_proporcion.groupby([bias_attr, target_attr]):
        print(f"Grupo ({bias_value}, {target_value}): {len(group)} muestras")

print("Cantidad de rubies en desarrollo: ", len(desarrollo[desarrollo["Blond_Hair"] == 1]))

