import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig
from google.api_core import exceptions
from dotenv import load_dotenv

# --- Cargar variables de entorno desde el archivo .env ---
# Es una mejor práctica de seguridad no tener la clave API directamente en el código.
load_dotenv()

app = Flask(__name__)

# --- Inicialización del Cliente de Gemini ---
try:
    # La clave API se lee automáticamente de la variable de entorno.
    # Asegúrate de que tu archivo .env tenga la línea: GEMINI_API_KEY="TU_CLAVE_API_AQUI"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
        # Asignamos None para manejar el error en las rutas
        model = None
    else:
        genai.configure(api_key=api_key)
        # CORRECCIÓN: Se instancia el modelo específico que se va a usar.
        # El system_instruction se define al crear el modelo.
        system_instruction = "Eres un asistente virtual amigable, útil y conciso. Responde siempre en español."
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_instruction
        )
        print("Modelo de Gemini inicializado con éxito.")

except Exception as e:
    print(f"Error al inicializar el cliente de Gemini: {e}")
    model = None

# --- Rutas de la Aplicación ---

@app.route("/")
def index():
    # Renderiza el archivo index.html que estará en la carpeta 'templates'
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    # 1. Verificar si el modelo se inicializó correctamente
    if not model:
        respuesta = "Lo siento, el servicio de IA no está configurado correctamente. Revisa la clave API en el servidor."
        return jsonify({"respuesta": respuesta}), 500

    # 2. Obtener el mensaje del usuario del cuerpo del request
    data = request.get_json()
    user_message = data.get("mensaje", "")

    # 3. Manejar el caso de mensaje vacío
    if not user_message.strip():
        return jsonify({"respuesta": "Por favor, escribe un mensaje."})

    try:
        # 4. Llamada a la API de Gemini (usando el modelo ya instanciado)
        # CORRECCIÓN: La forma de llamar a la API ha sido actualizada a la sintaxis correcta.
        # El contenido se pasa como una lista.
        response = model.generate_content([user_message])
        
        # 5. Extraer el texto de la respuesta
        # CORRECCIÓN: Se accede al texto a través de `response.text`
        respuesta = response.text.strip()

    except exceptions.GoogleAPICallError as e:
        print(f"Error de la API de Gemini: {e}")
        respuesta = "Hubo un problema al contactar la API de Gemini. Inténtalo de nuevo más tarde."
    
    except Exception as e:
        print(f"Error inesperado al generar contenido: {e}")
        respuesta = "Lo siento, ocurrió un error interno inesperado."

    return jsonify({"respuesta": respuesta})

# --- Ejecución ---
if __name__ == "__main__":
    # El puerto 5000 es el predeterminado para Flask
    app.run(debug=True, port=5000)