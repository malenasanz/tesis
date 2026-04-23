from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.datasets import CelebADataSet
import pandas as pd


def separar_desarrollo_test(df, bias_attr, target_attr, test_size=0.1, seed=42):
    grupos = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
              for b in [0, 1] for t in [0, 1]}
    n = min(len(g) for g in grupos.values())
    n_test = int(test_size * n)

    desarrollo_dfs, test_dfs = [], []
    for (b, t), g in grupos.items():
        g = g.sample(frac=1, random_state=seed)
        test_dfs.append(g.iloc[:n_test])
        desarrollo_dfs.append(g.iloc[n_test:n])

    desarrollo_df = pd.concat(desarrollo_dfs)
    test_df       = pd.concat(test_dfs)

    print(f"Desarrollo: {len(desarrollo_df)} muestras ({n - n_test} por subgrupo)")
    print(f"Test:       {len(test_df)} muestras ({n_test} por subgrupo)")
    for (b, t), g in test_df.groupby([bias_attr, target_attr]):
        print(f"  test ({b},{t}): {len(g)}")

    return desarrollo_df, test_df


def get_subset_con_proporciones(df, bias_attr, target_attr, p1, p2, val_size=0.1, seed=42):
    """
    p1: proporcion de bias=1 dentro de target=1
    p2: proporcion de bias=1 dentro de target=0
    """
    N = len(df) // 2
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

    train_df = pd.concat(train_dfs)
    val_df   = pd.concat(val_dfs)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | p1={p1:.2f} p2={p2:.2f}")
    for target_val, group in train_df.groupby(target_attr):
        total = len(group)
        for bias_val, subgroup in group.groupby(bias_attr):
            print(f"  train target={target_val}, bias={bias_val}: {len(subgroup)/total*100:.1f}% ({len(subgroup)})")

    return train_df, val_df


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataSet(
        img_dir=config["img_dir"],
        attr_path=config["attr_path"],
        transform=transform,
        target_attr=config["target_attr"]
    )

    seed       = config.get("seed", 42)
    bias_attr  = config["bias_attr"]
    target_attr = config["target_attr"]
    name_to_idx = {name: i for i, name in enumerate(dataset.images)}

    desarrollo_df, test_df = separar_desarrollo_test(
        dataset.df, bias_attr, target_attr,
        test_size=config.get("test_size", 0.1),
        seed=seed
    )

    train_df, val_df = get_subset_con_proporciones(
        desarrollo_df, bias_attr, target_attr,
        p1=config.get("p1", 0.5),
        p2=config.get("p2", 0.5),
        val_size=config.get("val_size", 0.1),
        seed=seed
    )

    train_dataset = Subset(dataset, [name_to_idx[n] for n in train_df.index])
    val_dataset   = Subset(dataset, [name_to_idx[n] for n in val_df.index])
    test_dataset  = Subset(dataset, [name_to_idx[n] for n in test_df.index])

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, val_loader, test_loader, dataset, test_df
