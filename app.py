from flask import Flask, jsonify, request, send_from_directory
import os

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Serve the index.html in the same folder
    root = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root, 'index.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json() or {}
    user_message = (data.get('mensaje', '') or '').lower()

    if 'hola' in user_message:
        respuesta = 'Hola'
    elif 'como estas' in user_message:
        respuesta = 'Bello'
    elif 'nombre' in user_message:
        respuesta = 'Soy Miguel'
    elif 'adios' in user_message:
        respuesta = 'Hasta luego'
    else:
        respuesta = 'Lo siento no entendi'

    return jsonify({'respuesta': respuesta})


if __name__ == '__main__':
    # Run on port 5000 by default
    app.run(host='127.0.0.1', port=5000, debug=True)