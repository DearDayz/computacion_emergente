from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import glob
import json

app = Flask(__name__, static_folder='.', static_url_path='')

MODEL_PATH = "surprise_vs_angry.keras"
EXAMPLES_DIR = "examples"
CLASS_NAMES_FILE = os.path.join(EXAMPLES_DIR, "class_names.json")


def find_latest_checkpoint():
    files = glob.glob(os.path.join("checkpoints", "save_at_*.keras"))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


# Cargar modelo (intenta archivo principal, si no existe usa último checkpoint)
model_path_to_load = None
if os.path.exists(MODEL_PATH):
    model_path_to_load = MODEL_PATH
else:
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        model_path_to_load = latest_ckpt
        print(f"No se encontró {MODEL_PATH}, usando checkpoint {latest_ckpt}")
    else:
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH} y no hay checkpoints save_at_*.keras. Ejecuta main.py para entrenarlo.")

print("Cargando modelo desde:", model_path_to_load)
model = keras.models.load_model(model_path_to_load)
# si cargamos un checkpoint y queremos una copia estándar
if model_path_to_load != MODEL_PATH:
    try:
        model.save(MODEL_PATH)
        print(f"Modelo guardado como {MODEL_PATH}")
    except Exception as e:
        print("Advertencia: no se pudo guardar copia estándar del modelo:", e)

# inferir tamaño de entrada desde el modelo cargado
try:
    input_shape = model.input_shape  # (None, H, W, C)
    IMAGE_SIZE = (int(input_shape[1]), int(input_shape[2]))
except Exception:
    IMAGE_SIZE = (180, 180)

# Cargar nombres de clase si existe
if os.path.exists(CLASS_NAMES_FILE):
    try:
        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            CLASS_NAMES = data.get("class_names", ["angry", "surprise"])
    except Exception:
        CLASS_NAMES = ["angry", "surprise"]
else:
    CLASS_NAMES = ["angry", "surprise"]


def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # batch
    return arr


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió archivo 'image'"}), 400
    file = request.files["image"]
    try:
        img_arr = preprocess_image(file.stream)
    except Exception as e:
        return jsonify({"error": f"No se pudo procesar la imagen: {e}"}), 400

    preds = model.predict(img_arr)
    # Manejar salida binaria (units==1) o multiclass
    if preds.shape[-1] == 1:
        # logits -> sigmoid
        prob_surprise = float(1.0 / (1.0 + np.exp(-preds[0][0])))
        prob_angry = 1.0 - prob_surprise
        probs = {CLASS_NAMES[0]: prob_angry, CLASS_NAMES[1]: prob_surprise}
        predicted = CLASS_NAMES[1] if prob_surprise >= 0.5 else CLASS_NAMES[0]
    else:
        exps = np.exp(preds[0] - np.max(preds[0]))
        soft = exps / exps.sum()
        probs = {CLASS_NAMES[i]: float(soft[i]) for i in range(len(soft))}
        predicted = max(probs, key=probs.get)

    return jsonify({"predicted": predicted, "probabilities": probs})


if __name__ == "__main__":
    # Ejecutar en localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=True)