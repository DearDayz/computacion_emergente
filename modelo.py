import google.generativeai as genai
import os

# Configura la API key desde variable de entorno para evitar exponerla en código.
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    raise RuntimeError("La variable de entorno GEMINI_API_KEY no está definida.")

def list_models():
    for model in genai.list_models():
        print(f"Nombre: {model.name}")
        print(f"Descripción: {getattr(model, 'description', 'Sin descripción')}")
        print(f"Acciones soportadas: {model.supported_generation_methods}")
        print("------")

if __name__ == "__main__":
    list_models()