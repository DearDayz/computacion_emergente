import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt


# Filtrar imágenes corruptas
num_skipped = 0
for folder_name in ("sad", "happy"):
    folder_path = os.path.join("emotions", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
            fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
        except Exception as e:
            print(f"Error con el archivo {fpath}: {e}")

print(f"Se eliminaron {num_skipped} imágenes corruptas.")

# Parámetros
image_size = (180, 180)
batch_size = 128

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
print("Clases:", class_names)

# Visualizar algunas imágenes del dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Definir data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Visualizar ejemplos aumentados
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[i]).astype("uint8"))
        plt.axis("off")
plt.show()

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

model = make_model(input_shape=image_size + (3,), num_classes=len(class_names))
model.summary()

keras.utils.plot_model(model, show_shapes=True, to_file="model_plot.png")

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# Entrenamiento
epochs = 25
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Ejemplo de predicción
img_path = "emotions/sad_00000.jpg"
img = keras.utils.load_img(img_path, target_size=image_size)
plt.imshow(img)
plt.axis("off")
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, 0)  # Añadir batch axis

predictions = model.predict(img_array)
score = float(keras.activations.sigmoid(predictions[0][0]))

print(f"Esta imagen es {100 * (1 - score):.2f}% triste y {100 * score:.2f}% feliz.")
