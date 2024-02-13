from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

app.secret_key = os.urandom(24).hex()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

data = {
    'Value': [0, 1, 2],
    'Sentiment': ['Negative', 'Neutral', 'Positive']
}

# Create DataFrame
dr = pd.DataFrame(data)

class_names=dr.Sentiment.tolist()

checkpoint_path = "best_model_USE"
# Load the entire model
model_1 = load_model(checkpoint_path)

def random_predictions_single(model, input_text, class_names=None):
    # Making predictions on the single input
    input_text = np.array([input_text])  # Convert to a numpy array for consistency
    y_pred_probs = model.predict(input_text)

    num_classes = y_pred_probs.shape[1] if len(y_pred_probs.shape) > 1 else 2
    is_binary_classification = num_classes == 2

    if is_binary_classification:
        y_pred = np.squeeze(np.round(y_pred_probs).astype(int))
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    # If class names are provided, use them for printing
    if class_names is not None:
        predicted_label_name = class_names[y_pred[0]]
    else:
        predicted_label_name = y_pred[0]

    # Print the predicted label
    print(f"Text: {input_text[0]}")
    print(f"Predicted: {predicted_label_name}")
    return input_text[0], predicted_label_name

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        text_inp = request.form['text_inp']
        input_text, sentiment = random_predictions_single(model_1, text_inp, class_names=class_names)
        return render_template('index.html', input_text=input_text, sentiment=sentiment  )
    return render_template('index.html')