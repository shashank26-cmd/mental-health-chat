from flask import Flask, request, render_template, jsonify
import numpy as np
import json
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the intents file
with open('artifacts/intents.json') as file:
    data = json.load(file)

# Load trained model
model = tf.keras.models.load_model('artifacts/best_model.h5')

# Load tokenizer object
with open('artifacts/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('artifacts/label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

@app.route('/')
def home():
    return render_template('index.html', user_message=None, bot_response=None)  # Render the index.html form

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')  # Get message from the form

    if not user_message:
        return jsonify({"reply": "I didn't get that. Can you say it again?"})

    # Predict the response
    result = model.predict(
        pad_sequences(tokenizer.texts_to_sequences([user_message]), truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    # Search for the response in the intents
    for i in data['intents']:
        if i['tag'] == tag:
            bot_response = random.choice(i['responses'])
            break
    else:
        bot_response = "I didn't understand that. Can you please rephrase?"

    return render_template('index.html', user_message=user_message, bot_response=bot_response)  # Return the response

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
