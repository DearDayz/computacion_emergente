import glob
import os
from tensorflow import keras

ckpts = glob.glob(os.path.join("checkpoints", "save_at_*.keras"))
if not ckpts:
    print("No se encontraron checkpoints 'save_at_*.keras' en la carpeta. Ejecuta main.py para entrenar.")
    raise SystemExit(1)

ckpts.sort(key=os.path.getmtime)
latest = ckpts[-1]
print("Último checkpoint encontrado:", latest)

print("Cargando checkpoint...")
model = keras.models.load_model(latest)
print("Guardando como surprise_vs_angry.keras ...")
model.save("surprise_vs_angry.keras")
print("Hecho: surprise_vs_angry.keras creado.")
