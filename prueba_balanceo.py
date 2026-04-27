from src.dataloader import split_balanceado, get_subset_con_proporciones
import pandas as pd 

import pandas as pd

df = pd.read_csv(
    "data/raw/celeba/list_attr_celeba.txt",
    sep="\s+",
    skiprows=1
)
df = (df+1)//2

train, val, test = split_balanceado(df, "Male", "Blond_Hair")
train_proporcion, val_proporcion = get_subset_con_proporciones(train, val, "Male", "Blond_Hair", p1=0, p2=0.2)
