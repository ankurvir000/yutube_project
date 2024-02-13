
# Sentiment Analysis Model 

This repository contains code for a Sentiment Analysis model using TensorFlow and Scikit-Learn. The model predicts sentiment (negative, neutral, positive) based on text input.

# Requirements

The project requires the following libraries:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow 2.x
- Colorama

# Data Loading and Preprocessing

 data_loading_preprocessing:
  - Load dataset: 'dataset/comments.csv'
  - Preprocess text:
      - Clean text using regex for emojis, special characters, hashtags, etc.
      - Tokenization and padding sequences


# Model Architecture:

- Neural Network Architecture:
  - Embedding layer
   - LSTM/SimpleRNN layers
  - Dense layers with appropriate activation functions
  - Output layer (softmax for multi-class classification)

# Model Training

  - Split dataset into train and validation sets
  - Compile model: Loss function, optimizer, and evaluation metrics
  - Train model: Epochs, batch size, callbacks (ModelCheckpoint)
  
# Model Evaluation and Visualization

  - Evaluate model: Classification report, confusion matrix
  - Visualize predictions: Random samples and their predictions

Install necessary dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
venv\Scripts\Activate
set FLASK_APP=app.py
flask run