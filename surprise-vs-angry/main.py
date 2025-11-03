# --- QUICK MODE: ajusta para ejecución rápida en CPU (pegar en main.py) ---
QUICK = True  # poner False para entrenamiento normal

if QUICK:
    image_size = (96, 96)      # imagen más pequeña -> menos cómputo
    batch_size = 32            # tamaño de batch moderado en CPU
    epochs = 100                # entrenar más en modo rápido (puede tardar más)
    MAX_TRAIN_BATCHES = 80     # número de batches por epoch (80*32 ≈ 2560 imágenes)
    MAX_VAL_BATCHES = 15
    USE_AUGMENTATION = True    # activar aumento para mejorar generalización
    MAKE_TINY_MODEL = True     # usar un modelo pequeño y rápido
else:
    image_size = (180, 180)
    batch_size = 128
    epochs = 25
    USE_AUGMENTATION = True
    MAKE_TINY_MODEL = False
# ------------------------------------------------------------------------

import os
import json
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib
# usar backend sin interfaz gráfica para poder guardar imágenes a disco
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import time

# Filtrar imágenes corruptas (usando Pillow.verify)
num_skipped = 0
for folder_name in ("surprise", "angry"):
    folder_path = os.path.join("emotions", folder_name)
    if not os.path.exists(folder_path):
        print(f"Advertencia: No se encontró la carpeta {folder_path}. Saltando...")
        continue

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as im:
                im.verify()  # lanzará excepción si está corrupta
        except Exception as e:
            num_skipped += 1
            try:
                os.remove(fpath)
                print(f"Eliminada imagen corrupta: {fpath}")
            except Exception:
                print(f"No se pudo eliminar: {fpath}")
print(f"Se eliminaron {num_skipped} imágenes corruptas.")

# Parámetros
# image_size = (180, 180)
# batch_size = 128

# Crear datasets de entrenamiento y validación
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "emotions",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
class_names = train_ds.class_names
# después de crear train_ds, val_ds con image_dataset_from_directory
if QUICK:
    # limitar número de batches para acelerar pruebas en CPU
    train_ds = train_ds.take(MAX_TRAIN_BATCHES)
    val_ds = val_ds.take(MAX_VAL_BATCHES)
print("Clases encontradas:", class_names)  # e.g. ['angry', 'surprise']

# Crear carpeta para ejemplos/artefactos que el frontend/servidor pueda usar
examples_dir = "examples"
os.makedirs(examples_dir, exist_ok=True)

# Visualizar algunas imágenes del dataset -> guardar a disco (no abrir ventana)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(examples_dir, "train_samples.png"))
plt.close()

# Definir data augmentation
if not USE_AUGMENTATION:
    def data_augmentation(images):
        return images
else:
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

# Visualizar ejemplos aumentados -> guardar a disco
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    augmented_images = data_augmentation(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[i]).astype("uint8"))
        plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(examples_dir, "augmented_samples.png"))
plt.close()

# Aplicar data augmentation de forma asincrónica
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x), y),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# Construir modelo
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

def make_tiny_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    if num_classes == 2:
        outputs = layers.Dense(1, activation=None)(x)
    else:
        outputs = layers.Dense(num_classes, activation=None)(x)
    return keras.Model(inputs, outputs)

if MAKE_TINY_MODEL:
    model = make_tiny_model(input_shape=image_size + (3,), num_classes=len(class_names))
else:
    model = make_model(input_shape=image_size + (3,), num_classes=len(class_names))  # tu modelo grande existente
model.summary()

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# Entrenamiento
callbacks = [
    # guardar checkpoints por epoch dentro de la carpeta checkpoints/
    keras.callbacks.ModelCheckpoint(os.path.join("checkpoints", "save_at_{epoch}.keras")),
]

# crear carpeta de checkpoints y añadir callbacks más útiles en QUICK: guardar el mejor y early stopping
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)
callbacks.append(keras.callbacks.ModelCheckpoint("surprise_vs_angry_best.keras", save_best_only=True, verbose=1))
# EarlyStopping: aumentar patience y mostrar mensaje cuando ocurra
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1))

# Reducir LR si la validación no mejora (mostrar mensajes)
callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1))

# calcular class_weight automáticamente (por si alguna clase queda ligeramente desbalanceada)
def compute_class_counts(folder="emotions"):
    counts = {}
    for cls in os.listdir(folder):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            continue
        n = sum(1 for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        counts[cls] = n
    return counts

counts = compute_class_counts("emotions")
print("Image counts per class:", counts)
if counts:
    # map class_names to counts (image_dataset_from_directory ordena alfabéticamente)
    weights = {}
    total = sum(counts.get(c, 0) for c in class_names)
    for i, cname in enumerate(class_names):
        cnt = counts.get(cname, 0)
        # evitar división por cero
        weights[i] = (total / (cnt + 1)) if cnt > 0 else 1.0
    print("class_weight:", weights)
else:
    weights = None

t0 = time.time()
if weights:
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, class_weight=weights)
else:
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
print(f"Entrenamiento completo en {time.time()-t0:.1f} segundos")

model.save("surprise_vs_angry.keras")
print("Modelo guardado en surprise_vs_angry.keras")

# Guardar nombres de clase para que el servidor/frontend los lea
with open(os.path.join(examples_dir, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

# Ejemplo de predicción - elegir imagen de ejemplo automáticamente si no existe la ruta fija
def find_example_image():
    # intentar imagen fija primero
    fixed = os.path.join("emotions", "angry", "angry_aug01040.jpg")
    if os.path.exists(fixed):
        return fixed
    # buscar cualquier imagen en emotions/<class>/*
    for cls in ("angry", "surprise"):
        folder = os.path.join("emotions", cls)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(folder, fname)
    return None

img_path = find_example_image()
if img_path is None:
    print("No se encontró imagen de ejemplo en 'emotions/'. Saltando predicción de ejemplo.")
else:
    try:
        img = keras.utils.load_img(img_path, target_size=image_size)
        # guardar la imagen de ejemplo que se usó
        example_input_path = os.path.join(examples_dir, "example_input.png")
        img.save(example_input_path)

        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)  # Añadir batch axis

        predictions = model.predict(img_array)
        # Si salida es única (units==1) -> usar sigmoid
        if predictions.shape[-1] == 1:
            raw_score = float(1.0 / (1.0 + np.exp(-predictions[0][0])))  # sigmoid
            probs = {
                class_names[0]: 1.0 - raw_score,
                class_names[1]: raw_score
            }
        else:
            # multi-clase
            exps = np.exp(predictions[0] - np.max(predictions[0]))
            soft = exps / exps.sum()
            probs = {class_names[i]: float(soft[i]) for i in range(len(soft))}
            raw_score = None

        result = {
            "image": example_input_path,
            "probabilities": probs,
            "raw_score": raw_score,
        }

        with open(os.path.join(examples_dir, "example_prediction.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Predicción de ejemplo guardada en {os.path.join(examples_dir, 'example_prediction.json')}")
    except Exception as e:
        print(f"Error haciendo predicción de ejemplo con {img_path}: {e}")