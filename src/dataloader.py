from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from .datasets import CelebADataSet
import pandas as pd
import numpy as np



def split_balanceado(df, bias_attr, target_attr, seed=42):
    grupos = {(b, t): df[(df[bias_attr] == b) & (df[target_attr] == t)]
              for b in [0, 1] for t in [0, 1]}
    n = min(len(g) for g in grupos.values())

    train, val, test = [], [], []
    for (b, t), g in grupos.items():
        g = g.sample(n=n, random_state=seed)
        train_idx = np.random.choice(g.index, size= int(0.8*n), replace=False)
        train.append(train_idx)
        val_idx = np.random.choice(list(set(g.index) - set(train_idx)), size= int(0.1*n), replace=False)
        val.append(val_idx)
        test_idx = list(set(g.index) - set(train_idx) - set(val_idx))
        test.append(test_idx)

    print(f"Train: {int(0.8*n*4)} muestras")
    for (b, t), idxs in zip([(b, t) for b in [0, 1] for t in [0, 1]], train):
        print(f"  train ({b},{t}): {len(idxs)}")
    print(f"Val:        {int(0.1*n*4)} muestras")
    for (b, t), idxs in zip([(b, t) for b in [0, 1] for t in [0, 1]], val):
        print(f"  val ({b},{t}): {len(idxs)}")
    print(f"Test:       {int(0.1*n*4)} muestras")
    for (b, t), idxs in zip([(b, t) for b in [0, 1] for t in [0, 1]], test):
        print(f"  test ({b},{t}): {len(idxs )}")

    return train, val, test


def get_subset_con_proporciones(train_total, val_total, bias_attr, target_attr, p1, p2, val_size=0.1, seed=42):
    """
    p1: proporcion de bias=1 dentro de target=0 (hombres no rubios)
    p2: proporcion de bias=1 dentro de target=1 (hombres rubios)
    """
    n_train = len(train_total[0])
    n_val = len(val_total[0])

    counts_train = [(1-p1)*n_train, (1-p2)*n_train, p1*n_train, p2*n_train]
    counts_val   = [(1-p1)*n_val,   (1-p2)*n_val,   p1*n_val,   p2*n_val]

    train, val = [], []

    for i in range(len(train_total)):
        train_idx = np.random.choice(train_total[i], size=int(counts_train[i]))
        train.append(train_idx)
        val_idx = np.random.choice(val_total[i], size=int(counts_val[i]))
        val.append(val_idx)

    print(f"Train: {sum(len(x) for x in train)} muestras. Coincide con {n_train*2}")
    print(f"Val: {sum(len(x) for x in val)} muestras. Coincide con {n_val*2}")
    keys = [(b, t) for b in [0, 1] for t in [0, 1]]
    for i, (b, t) in enumerate(keys):
        b_label = bias_attr if b == 1 else f"no {bias_attr}"
        t_label = target_attr if t == 1 else f"no {target_attr}"
        print(f"  {b_label}, {t_label}: {len(train[i])*100/n_train:.2f}% ({len(train[i])} muestras)")
    return train, val


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

    seed        = config.get("seed", 42)
    bias_attr   = config["bias_attr"]
    target_attr = config["target_attr"]
    name_to_idx = {name: i for i, name in enumerate(dataset.images)}

    train_bal, val_bal, test_bal = split_balanceado(dataset.df, bias_attr, target_attr, seed)

    train_idx, val_idx = get_subset_con_proporciones(
        train_bal, val_bal, bias_attr, target_attr,
        p1=config.get("p1", 0.5),
        p2=config.get("p2", 0.5),
        seed=seed
    )

    all_train = np.concatenate(train_idx)
    all_val   = np.concatenate(val_idx)
    all_test  = np.concatenate(test_bal)

    overlap_train_test = len(set(all_train) & set(all_test))
    overlap_train_val  = len(set(all_train) & set(all_val))
    overlap_val_test   = len(set(all_val)   & set(all_test))
    print(f"Overlap train-test: {overlap_train_test} | train-val: {overlap_train_val} | val-test: {overlap_val_test}")

    test_df = dataset.df.loc[all_test]

    train_dataset = Subset(dataset, [name_to_idx[n] for n in all_train])
    val_dataset   = Subset(dataset, [name_to_idx[n] for n in all_val])
    test_dataset  = Subset(dataset, [name_to_idx[n] for n in all_test])

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, val_loader, test_loader, dataset, test_df


def get_dataloaders_simple(config):
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

    seed = config.get("seed", 42)
    rng  = np.random.default_rng(seed)

    n       = len(dataset)
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"Simple split — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size)

    test_df = dataset.df.iloc[test_idx]

    return train_loader, val_loader, test_loader, dataset, test_df
