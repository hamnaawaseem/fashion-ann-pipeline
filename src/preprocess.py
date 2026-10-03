# TODO: try different validation split

import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

train = np.load("data/raw/train.npz")
test = np.load("data/raw/test.npz")

x_train = (train["x"].astype("float32") - 127.5) / 127.5
x_test = (test["x"].astype("float32") - 127.5) / 127.5

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, train["y"],
    test_size=params["val_split"],
    random_state=params["seed"],
    stratify=train["y"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
np.savez_compressed("data/processed/test.npz", x=x_test, y=test["y"])

print(f"train {x_tr.shape}, val {x_val.shape}, test {x_test.shape}")