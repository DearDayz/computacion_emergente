import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDazOXqGwABHUE_k6qfLPiz9iGUQo41Y8Y")

def list_models():
    for model in genai.list_models():
        print(f"Nombre: {model.name}")
        print(f"Descripción: {getattr(model, 'description', 'Sin descripción')}")
        print(f"Acciones soportadas: {model.supported_generation_methods}")
        print("------")

if __name__ == "__main__":
    list_models()