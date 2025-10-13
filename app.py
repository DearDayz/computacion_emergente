import os
from flask import Flask, request, jsonify, render_template
from google.api_core import exceptions
from dotenv import load_dotenv

# La doc oficial importa así el genai, por lo que tuve que adaptar el resto de cosas
from google import genai

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
        client = None
    else:
        client = genai.Client(api_key=api_key)
        print("Modelo de Gemini inicializado con éxito.")

except Exception as e:
    print(f"Error al inicializar el cliente de Gemini: {e}")
    client = None

# --- Rutas de la Aplicación ---

@app.route("/")
def index():
    # Renderiza el archivo index.html que estará en la carpeta 'templates'
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    # 1. Verificar si el modelo se inicializó correctamente
    if not client:
        respuesta = "Lo siento, el servicio de IA no está configurado correctamente. Revisa la clave API en el servidor."
        return jsonify({"respuesta": respuesta}), 500

    # 2. Obtener el mensaje del usuario del cuerpo del request
    data = request.get_json()
    user_message = data.get("mensaje", "")

    # 3. Manejar el caso de mensaje vacío
    if not user_message.strip():
        return jsonify({"respuesta": "Por favor, escribe un mensaje."})

    try:
        # 4. Llamada a la API de Gemini
        system_instruction = "Eres un asistente virtual amigable, útil y conciso. Responde siempre en español."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[f"{system_instruction}\nUsuario: {user_message}\nAsistente:"],
        )

        # 5. Extraer el texto de la respuesta
        respuesta = response.text.strip()

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
        # Guardar el archivo temporalmente
        temp_audio_path = "temp_audio.mp3"
        audio_file.save(temp_audio_path)

        # Subir el archivo a la API de Gemini
        myfile = client.files.upload(file=temp_audio_path)

        # Crear el prompt para la transcripción
        prompt = "Genera una transcripción del contenido del audio."

        # Llamar al modelo de Gemini para procesar el archivo
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, myfile]
        )

        # Extraer la respuesta
        respuesta = response.text.strip()

    except exceptions.GoogleAPICallError as e:
        print(f"Error de la API de Gemini: {e}")
        respuesta = "Hubo un problema al contactar la API de Gemini. Inténtalo de nuevo más tarde."
    
    except Exception as e:
        print(f"Error inesperado al procesar el archivo de audio: {e}")
        respuesta = "Lo siento, ocurrió un error interno inesperado."

    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return jsonify({"respuesta": respuesta})

# --- Ejecución ---
if __name__ == "__main__":
    # El puerto 5000 es el predeterminado para Flask
    app.run(debug=True, port=5000)