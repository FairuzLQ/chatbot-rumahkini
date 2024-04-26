from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

@app.route("/")
def index_get():
    # Ensure that you have an 'index.html' in your templates folder
    return "Connected"  # or return "Hello, world!" for a simple text string

@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    print("Server is running...")
    app.run(debug=True)
