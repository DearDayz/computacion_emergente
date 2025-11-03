import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace

file_path = os.path.join(os.path.dirname(__file__), "10k_Poplar_Tv_Shows.csv")
dataframe = pd.read_csv(file_path)

TARGET_FEATURE_NAME = "high_rating"

print(dataframe[TARGET_FEATURE_NAME].value_counts())
print("Forma del DataFrame:", dataframe.shape)
print("Primeras 5 filas:")
print(dataframe.head())

# Como no tenemos una columna binaria vamos a hacerla a partir de 'vote_average'
# Si vote_average >= 8.0 entonces high_rating = 1, si no 0
# Esto suponiendo que nunca pasen de 10, (creo que es a lo que llega el dataset)
dataframe[TARGET_FEATURE_NAME] = (dataframe["vote_average"] >= 8.0).astype(int)

# Aqui definimos las columnas que vamos a usar en el entrenamiento
FEATURE_NAMES = [
    "popularity",
    "vote_count",
    "original_language",
    "origin_country",
    "original_name",
]

# Esto lo hacemos para eliminar las columnas que no usaremos
# Normalmente se puede hacer más adelante pero tiraba error si las dejaba
all_cols_to_keep = FEATURE_NAMES + [TARGET_FEATURE_NAME]
dataframe = dataframe[all_cols_to_keep] # Esto eliminará las columnas no listadas

print("Forma del DataFrame después de la limpieza y preprocesamiento:", dataframe.shape)
print("Columnas restantes:", dataframe.columns.tolist())

# División de datos (20% para validación)
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(f"Usando {len(train_dataframe)} muestras para entrenamiento y {len(val_dataframe)} para validación")

def dataframe_to_dataset(dataframe):
    # Convertir DataFrame de pandas a Dataset de TensorFlow
    dataframe = dataframe.copy()
    labels = dataframe.pop(TARGET_FEATURE_NAME)

    # Al hacer esta conversión era que arrojaba el error con las columnas sobrantes
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

feature_space = FeatureSpace(
    # Definición de features o columnas a usar
    features={
        "popularity": FeatureSpace.float_normalized(),
        "vote_count": FeatureSpace.float_normalized(),
        "original_language": FeatureSpace.string_categorical(num_oov_indices=1),
        "origin_country": FeatureSpace.string_categorical(num_oov_indices=1),
        "original_name": FeatureSpace.string_categorical(num_oov_indices=1),
    },
    output_mode="concat",
)

# --- Adaptación y Preprocesamiento ---

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

# --- Construcción, Compilación y Entrenamiento del Modelo---

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

print("\n--- Entrenamiento del Modelo ---")
training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
    verbose=2,
)

# Guardamos el modelo para no entrenarlo cada vez que se pruebe

script_dir = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(script_dir, "model.keras")

inference_model.save(save_path) 

print(f"\nModelo guardado correctamente en: {save_path}")