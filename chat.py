import random
import json
import torch
import time
from datetime import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

# Model parameters and state
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize and load the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = ""

def get_response(user_input):
    start_time = time.time()
    
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    response = "Mohon ulangi dan lengkapi pertanyaan yang kamu tanyakan :)"
    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                break
    
    end_time = time.time()
    response_time = end_time - start_time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_json = {
        "tag": tag,
        "response": response,
        "time_taken_to_respond": response_time,
        "date_asked": current_date,
        "question": user_input,
        "confidence": prob.item()
    }
    
    return output_json

# Example usage
# user_input = "Hello, how are you?"
# print(json.dumps(get_response(user_input), indent=4))
