import os
import logging
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from google.api_core import exceptions
from dotenv import load_dotenv
from google import genai

# --- Configuración de logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuración ---
class Config:
    """Clase de configuración para la aplicación"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MAX_MESSAGE_LENGTH = 1000
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg']
    UPLOAD_FOLDER = tempfile.gettempdir()

# --- Cargar variables de entorno ---
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)

# --- Inicialización del Cliente de Gemini ---
def initialize_gemini_client():
    """Inicializa el cliente de Gemini con manejo de errores"""
    try:
        api_key = app.config['GEMINI_API_KEY']
        if not api_key:
            logger.error("GEMINI_API_KEY no está configurada en variables de entorno")
            return None
        
        client = genai.Client(api_key=api_key)
        logger.info("Cliente de Gemini inicializado exitosamente")
        return client
    
    except Exception as e:
        logger.error(f"Error al inicializar cliente de Gemini: {str(e)}")
        return None

# Inicializar cliente global
client = initialize_gemini_client()

# --- Utilidades ---
def validate_message(message):
    """Valida el mensaje del usuario"""
    if not message or not isinstance(message, str):
        return False, "Mensaje no válido"
    
    if len(message.strip()) == 0:
        return False, "El mensaje no puede estar vacío"
    
    if len(message) > app.config['MAX_MESSAGE_LENGTH']:
        return False, f"El mensaje no puede exceder {app.config['MAX_MESSAGE_LENGTH']} caracteres"
    
    return True, "Mensaje válido"

def handle_api_error(error):
    """Maneja errores de la API de manera uniforme"""
    if isinstance(error, exceptions.GoogleAPICallError):
        logger.error(f"Error de API de Gemini: {str(error)}")
        return "Lo siento, hay un problema temporal con el servicio. Inténtalo de nuevo.", 503
    
    logger.error(f"Error inesperado: {str(error)}")
    return "Ha ocurrido un error interno. Por favor, inténtalo de nuevo.", 500

# --- Rutas de la Aplicación ---

@app.route("/")
def index():
    """Página principal del chatbot"""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar el estado de la aplicación"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gemini_client": "connected" if client else "disconnected"
    }
    
    return jsonify(status), 200 if client else 503

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Endpoint principal del chatbot"""
    # Verificar si el cliente está disponible
    if not client:
        logger.warning("Intento de usar chatbot sin cliente disponible")
        return jsonify({
            "respuesta": "El servicio de IA no está disponible. Verifica la configuración."
        }), 503

    try:
        # Obtener y validar datos de entrada
        if not request.is_json:
            return jsonify({"respuesta": "Content-Type debe ser application/json"}), 400
        
        data = request.get_json()
        user_message = data.get("mensaje", "")
        
        # Validar mensaje
        is_valid, validation_message = validate_message(user_message)
        if not is_valid:
            return jsonify({"respuesta": validation_message}), 400

        # Log del mensaje recibido
        logger.info(f"Mensaje recibido: {user_message[:50]}...")
        
        # Configurar prompt del sistema
        system_instruction = """Eres un asistente virtual amigable y útil llamado MSN Bot. 
        Responde siempre en español de manera concisa pero amigable. 
        Mantén un tono conversacional similar al de los chats clásicos de MSN Messenger."""
        
        # Llamada a la API de Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[f"{system_instruction}\nUsuario: {user_message}\nMSN Bot:"],
        )

        # Extraer y procesar respuesta
        bot_response = response.text.strip()
        logger.info(f"Respuesta generada: {bot_response[:50]}...")

        return jsonify({"respuesta": bot_response})

    except Exception as e:
        error_message, status_code = handle_api_error(e)
        return jsonify({"respuesta": error_message}), status_code

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    """Endpoint para convertir audio a texto"""
    if not client:
        return jsonify({
            "respuesta": "El servicio de reconocimiento de voz no está disponible."
        }), 503

    # Validar archivo de audio
    if 'audio' not in request.files:
        return jsonify({"respuesta": "Se requiere un archivo de audio"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"respuesta": "No se seleccionó archivo de audio"}), 400

    temp_audio_path = None
    try:
        # Crear archivo temporal seguro
        temp_audio_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            f"audio_{datetime.utcnow().timestamp()}.mp3"
        )
        
        # Guardar archivo
        audio_file.save(temp_audio_path)
        logger.info(f"Audio guardado temporalmente: {temp_audio_path}")

        # Subir archivo a Gemini
        myfile = client.files.upload(file=temp_audio_path)
        
        # Procesar transcripción
        prompt = "Transcribe exactamente el contenido de este audio en español. Solo devuelve el texto transcrito sin comentarios adicionales."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, myfile]
        )

        transcription = response.text.strip()
        logger.info(f"Transcripción completada: {transcription[:50]}...")

        return jsonify({"respuesta": transcription})

    except Exception as e:
        error_message, status_code = handle_api_error(e)
        return jsonify({"respuesta": error_message}), status_code

    finally:
        # Limpiar archivo temporal
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Archivo temporal eliminado: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal: {str(e)}")

# --- Manejo de Errores ---

@app.errorhandler(404)
def not_found(error):
    """Manejo de errores 404"""
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejo de errores internos"""
    logger.error(f"Error interno del servidor: {str(error)}")
    return jsonify({"error": "Error interno del servidor"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Manejo de archivos demasiado grandes"""
    return jsonify({"error": "El archivo es demasiado grande"}), 413

# --- Punto de entrada ---
if __name__ == "__main__":
    logger.info("Iniciando aplicación Flask...")
    
    # Verificar configuración antes de iniciar
    if not client:
        logger.warning("La aplicación se ejecutará sin cliente de Gemini")
    
    # Configuración de desarrollo vs producción
    debug_mode = os.getenv("FLASK_ENV") == "development"
    
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.getenv("PORT", 5000))
    )
