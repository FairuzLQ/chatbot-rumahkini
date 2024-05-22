from flask import Flask, render_template, request, jsonify
from chat import get_response
import logging

app = Flask(__name__)

logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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

if __name__ == "__main__":
    print("Server is running...")
    app.run(debug=True)
