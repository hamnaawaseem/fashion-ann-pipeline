import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow import keras

with open("params.yaml") as f:
    p = yaml.safe_load(f)["train"]

tf.random.set_seed(p["seed"])
np.random.seed(p["seed"])

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(p["units"], activation="relu"),
    keras.layers.Dropout(p["dropout"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train["x"], train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=p["epochs"],
    batch_size=p["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Model and history saved.")