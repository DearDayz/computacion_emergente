import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras

MODEL_NAME = "model.keras" 

# Se definen acá para garantizar que se le pasan las mismas features
# con las que fue entrenado el modelo
FEATURE_NAMES = [
    "popularity",
    "vote_count",
    "original_language",
    "origin_country",
    "original_name",
]

# --- Carga del modelo --- 

try:
    file_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)
    inference_model = keras.models.load_model(file_path)
    
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo. Asegúrate de que '{MODEL_NAME}' exista.")
    exit()

# --- Datos para la predicción ---

new_series_data = {
    "popularity": 10.0,
    "vote_count": 563,
    "original_language": "en",
    "origin_country": "VE",
    "original_name": "Invincible",
}

# Convertir los datos de la serie a un formato de Tensor compatible con Keras
input_dict = {
    name: tf.convert_to_tensor([value])
    for name, value in new_series_data.items()
    if name in FEATURE_NAMES # Esto garantiza la sincronización
}

predictions = inference_model.predict(input_dict)
probability = 100 * predictions[0][0]

print("\n--- Resultado de la Predicción ---")
print(f"Serie: {new_series_data['original_name']}")
print(f"Probabilidad de ser un título de ALTO RATING (> 8.0): **{probability:.2f}%**")