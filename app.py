import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Dataset path
DATASET_PATH = 'mail_data.csv'
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found!")

# Load dataset
raw_mail_data = pd.read_csv(DATASET_PATH)
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Ensure correct column names exist
if 'Category' not in mail_data.columns or 'Message' not in mail_data.columns:
    raise KeyError("Dataset must contain 'Category' and 'Message' columns.")

# Convert 'Category' column to binary values
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Splitting data
X = mail_data['Message']
Y = mail_data['Category'].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Function to predict spam
def predict_spam(text):
    text_features = vectorizer.transform([text])
    prediction = model.predict(text_features)[0]
    return "Not Spam" if prediction == 1 else "Spam"

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    result = predict_spam(message) if message else "No message provided"
    return render_template('index.html', prediction=result, message=message)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
