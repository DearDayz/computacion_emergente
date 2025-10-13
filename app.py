import os
from flask import Flask, request, jsonify, render_template
from google.api_core import exceptions
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tempfile
import time

# Usamos la librería instalada `google-generativeai`
import google.generativeai as genai

# --- Cargar variables de entorno desde el archivo .env ---
# Es una mejor práctica de seguridad no tener la clave API directamente en el código.
load_dotenv()

app = Flask(__name__)

# --- Inicialización del Cliente/Modelo de Gemini ---
try:
    # La clave API se lee automáticamente de la variable de entorno.
    # Asegúrate de que tu archivo .env tenga la línea: GEMINI_API_KEY="TU_CLAVE_API_AQUI"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
        # Asignamos None para manejar el error en las rutas
        model = None
    else:
        # Configurar la librería con la API key
        genai.configure(api_key=api_key)
        # Puedes ajustar el modelo aquí. Mantengo el id solicitado.
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction="Eres un asistente virtual amigable, útil y conciso. Responde siempre en español."
        )
        print("Gemini configurado e inicializado con éxito.")

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
        # 4. Llamada a la API de Gemini usando `google-generativeai`
        response = model.generate_content(
            [f"Usuario: {user_message}"]
        )

        # 5. Extraer el texto de la respuesta
        respuesta = (response.text or "").strip() or "No se recibió texto de respuesta."

    except exceptions.GoogleAPICallError as e:
        print(f"Error de la API de Gemini: {e}")
        respuesta = "Hubo un problema al contactar la API de Gemini. Inténtalo de nuevo más tarde."
    
    except Exception as e:
        print(f"Error inesperado al generar contenido: {e}")
        respuesta = "Lo siento, ocurrió un error interno inesperado."

    return jsonify({"respuesta": respuesta})

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"respuesta": "Faltan parámetros: se requiere un archivo de audio."}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({"respuesta": "No se seleccionó ningún archivo de audio."}), 400

    try:
        # Guardar el archivo temporalmente (simple y directo)
        # Nota: mantenemos .mp3 por compatibilidad con tu lógica anterior
        temp_audio_path = "temp_audio.mp3"
        audio_file.save(temp_audio_path)

        # Subir el archivo a la API de Gemini (flujo sencillo)
        uploaded = genai.upload_file(temp_audio_path)

        # Prompt para la transcripción
        prompt = "Genera una transcripción del contenido del audio."

        # Generar contenido con el archivo
        response = model.generate_content([prompt, uploaded])

        # Extraer la respuesta
        respuesta = (response.text or "").strip() or "No se pudo generar una transcripción."

    except exceptions.GoogleAPICallError as e:
        print(f"Error de la API de Gemini: {e}")
        respuesta = "Hubo un problema al contactar la API de Gemini. Inténtalo de nuevo más tarde."
    
    except Exception as e:
        print(f"Error inesperado al procesar el archivo de audio: {e}")
        respuesta = "Lo siento, ocurrió un error interno inesperado."

    finally:
        # Eliminar el archivo temporal
        if os.path.exists("temp_audio.mp3"):
            try:
                os.remove("temp_audio.mp3")
            except Exception:
                pass

    return jsonify({"respuesta": respuesta})

# --- Ejecución ---
if __name__ == "__main__":
    # El puerto 5000 es el predeterminado para Flask
    app.run(debug=True, port=5000)