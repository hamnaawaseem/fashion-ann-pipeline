import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

model = keras.models.load_model("models/model.h5")
test = np.load("data/processed/test.npz")

loss, acc = model.evaluate(test["x"], test["y"], verbose=0)

preds = np.argmax(model.predict(test["x"]), axis=1)
cm = confusion_matrix(test["y"], preds)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")