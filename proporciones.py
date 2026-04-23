import pandas as pd 

df = pd.read_csv(
    "data/raw/celeba/list_attr_celeba.txt",
    sep="\s+",
    skiprows=1
)

df = (df+1)//2

def separar_desarrollo_test(df, bias_attr, target_attr, split_size=0.1, seed=42):
    grupos = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
              for b in [0, 1] for t in [0, 1]}
    n = min(len(g) for g in grupos.values())
    n_split = int(split_size * n)

    main_dfs, split_dfs = [], []
    for g in grupos.values():
        g = g.sample(frac=1, random_state=seed)
        split_dfs.append(g.iloc[:n_split])
        main_dfs.append(g.iloc[n_split:n])

    main_df  = pd.concat(main_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    split_df = pd.concat(split_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    return main_df, split_df

def get_subset_con_proporciones(df, bias_attr, target_attr, p1, p2, val_size=0.1, seed=42):
    """
    p1: proporción de bias=1 dentro de target=1 (ej. hombres dentro de rubios)
    p2: proporción de bias=1 dentro de target=0 (ej. hombres dentro de no-rubios)
    """
    N = len(df) // 2 #tomo menos datos que el total, asi puedo variar las proporciones dejando la cantidad de datos fija
    g = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
         for b in [0, 1] for t in [0, 1]}

    counts = {
        (1, 1): int(p1 * N / 2),
        (0, 1): int((1 - p1) * N / 2),
        (1, 0): int(p2 * N / 2),
        (0, 0): int((1 - p2) * N / 2),
    }

    for key, n in counts.items():
        if n > len(g[key]):
            raise ValueError(f"Grupo {key}: se piden {n} pero solo hay {len(g[key])} disponibles.")

    train_dfs, val_dfs = [], []
    for key, n in counts.items():
        sampled = g[key].sample(n=n, random_state=seed)
        n_val = int(val_size * n)
        val_dfs.append(sampled.iloc[:n_val])
        train_dfs.append(sampled.iloc[n_val:])

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df

bias_attr = "Male"
target_attr = "Blond_Hair"
desarrollo, test = separar_desarrollo_test(df, bias_attr, target_attr, split_size=0.1) #cada uno balanceado
train_balanceado, val_balanceado =  get_subset_con_proporciones(desarrollo, bias_attr, target_attr, p1 = 0.5, p2 = 0.5)
train_no_balanceado, val_no_balanceado =  get_subset_con_proporciones(desarrollo, bias_attr, target_attr, p1 = 0.3, p2 = 0.5)

print("Cantidad datos test: ", len(test))
for (bias_value, target_value), group in test.groupby([bias_attr, target_attr]):
        print(f"Grupo ({bias_value}, {target_value}): {len(group)/len(test)*100:.2f}% muestras")

print("Cantidad datos train no balanceado: ", len(train_no_balanceado))
for target_value, group in train_no_balanceado.groupby(target_attr):
    total = len(group)
    for bias_value, subgroup in group.groupby(bias_attr):
        print(f"target={target_value}, bias={bias_value}: {len(subgroup)/total*100:.1f}%", f"({len(subgroup)}) muestras")


print(df.groupby([bias_attr, target_attr]).size())