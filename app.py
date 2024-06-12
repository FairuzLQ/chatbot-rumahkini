from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import logging
import time
from datetime import datetime, timedelta
import os
import nltk_utils

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')
start_time = time.time()

# Count the number of requests to the /predict endpoint
predict_request_count = 0

@app.before_request
def before_request():
    global predict_request_count
    if request.endpoint == 'predict':
        predict_request_count += 1

@app.route("/")
def index_get():
    return "Connected"

@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    if text:
        response = get_response(text)
        logging.info(f'User input: {text} - Chatbot response: {response}')
        response_with_status = {
            "status_code": 200,
            "data": response
        }
        return jsonify(response_with_status), 200
    else:
        response_with_status = {
            "status_code": 400,
            "error": "Mohon Ulangi Pertanyaan Anda :)"
        }
        logging.warning('No message found in request')
        return jsonify(response_with_status), 400

@app.route("/status", methods=['GET'])
def status():
    # Calculate uptime
    current_time = time.time()
    uptime_seconds = int(current_time - start_time)
    uptime_str = str(timedelta(seconds=uptime_seconds))

    # Get server time
    server_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    status_info = {
        "status": "Server is running",
        "version": "1.0.0",
        "uptime": uptime_str,
        "server_time": server_time,
        "predict_request_count": predict_request_count
    }
    return jsonify(status_info), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
