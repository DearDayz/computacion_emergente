"""Diagnóstico rápido: recorre las carpetas emotions/angry y emotions/surprise,
carga el modelo (o el checkpoint), y muestra logits/probabilidades por imagen
y una matriz de confusión simple.

Usar:
  python debug_predict.py
Opciones:
  --limit N    limitar a N imágenes por clase (útil para pruebas rápidas)
"""
import os
import glob
import json
import numpy as np
from tensorflow import keras
from PIL import Image
import argparse


def find_model():
    if os.path.exists("surprise_vs_angry.keras"):
        return "surprise_vs_angry.keras"
    ckpts = glob.glob(os.path.join("checkpoints", "save_at_*.keras"))
    if ckpts:
        ckpts.sort(key=os.path.getmtime)
        return ckpts[-1]
    return None


def load_class_names():
    p = os.path.join("examples", "class_names.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("class_names", ["angry", "surprise"])
        except Exception:
            pass
    return ["angry", "surprise"]


def preprocess_image(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    return arr


def main(limit=None):
    model_path = find_model()
    if model_path is None:
        print("No se encontró modelo ni checkpoints. Ejecuta main.py para entrenarlo.")
        return

    print("Cargando modelo:", model_path)
    model = keras.models.load_model(model_path)

    class_names = load_class_names()
    print("Class names:", class_names)

    # inferir image size
    try:
        in_shape = model.input_shape
        image_size = (int(in_shape[1]), int(in_shape[2]))
    except Exception:
        image_size = (180, 180)

    pairs = []  # list of (true_label, path)
    for cls in ("angry", "surprise"):
        folder = os.path.join("emotions", cls)
        if not os.path.exists(folder):
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        if limit:
            files = files[:limit]
        pairs += [(cls, p) for p in files]

    if not pairs:
        print("No se encontraron imágenes en emotions/angry ni emotions/surprise")
        return

    # stats
    correct = 0
    total = 0
    conf = {a: {b: 0 for b in ("angry", "surprise")} for a in ("angry", "surprise")}
    misclassified = []

    for true_label, path in pairs:
        arr = preprocess_image(path, image_size)
        preds = model.predict(arr)
        if preds.shape[-1] == 1:
            logit = float(preds[0][0])
            prob_surprise = 1.0 / (1.0 + np.exp(-logit))
            prob_angry = 1.0 - prob_surprise
            probs = {"angry": prob_angry, "surprise": prob_surprise}
            predicted = "surprise" if prob_surprise >= 0.5 else "angry"
        else:
            exps = np.exp(preds[0] - np.max(preds[0]))
            soft = exps / exps.sum()
            probs = {class_names[i]: float(soft[i]) for i in range(len(soft))}
            predicted = max(probs, key=probs.get)

        total += 1
        if predicted == true_label:
            correct += 1
        else:
            misclassified.append((path, true_label, predicted, probs))

        conf[true_label][predicted] += 1
        print(f"{os.path.basename(path)} -> pred={predicted} probs={probs}")

    print("\nSummary:")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.3f}")
    print("Confusion:")
    print(conf)
    if misclassified:
        print("\nSome misclassified examples:")
        for p, t, pr, probs in misclassified[:10]:
            print(f"{p} true={t} pred={pr} probs={probs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="max images per class to test")
    args = ap.parse_args()
    main(limit=args.limit)
