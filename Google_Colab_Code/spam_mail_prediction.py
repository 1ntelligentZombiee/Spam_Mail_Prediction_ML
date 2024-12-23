# -*- coding: utf-8 -*-
"""Spam Mail Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tgQtrE2ngGZixsrlNi2vqijYwoJ_EvL-
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import re
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse as sp

mail_data = pd.read_csv('/content/emails.csv')

# Rename 'spam' column to 'Label'
mail_data.rename(columns={'spam': 'Label'}, inplace=True)
mail_data.rename(columns={'text': 'Body'}, inplace=True)

mail_data.head()

mail_data.info()

# checking the number of rows and columns in the dataframe
mail_data.shape

mail_data.describe().T

len(mail_data['Body'].unique())

mail_data.isnull().sum()

mail_data.duplicated().sum()

duplicate=mail_data[mail_data.duplicated(keep='last')]
duplicate

counts = mail_data['Label'].value_counts().reset_index()
counts.columns = ['Label', 'Count']
# Create a bar plot using Plotly Express
fig = px.bar(counts, x='Label', y='Count', color='Label')
fig.update_layout(title='Number of Spam and Ham Emails', xaxis_title='Label', yaxis_title='Count')
fig.update_xaxes(tickvals=[0, 1], ticktext=['Ham', 'Spam'])
fig.show()

# separating the data as texts and label

X = mail_data['Body']

Y = mail_data['Label']

print(X)

print(Y)

# Check class distribution
print(mail_data['Label'].value_counts())

# Shuffle the dataset
mail_data = mail_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first few rows
print(mail_data.head())

print(mail_data['Label'].head(10))

# Check class distribution
print(mail_data['Label'].value_counts())

# Preprocessing Function
def preprocess_email(text):
    # Handle missing values
    if pd.isnull(text):
        text = ""

    # Count URLs
    num_urls = len(re.findall(r'(https?://\S+)', text))

    # Check for attachments (simple keyword-based detection)
    has_attachment = bool(re.search(r'attachment|attached|file', text, re.IGNORECASE))

    # Remove URLs
    text = re.sub(r'(https?://\S+)', '', text)

    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)

    # Lowercase all text
    text = text.lower()

    return text, num_urls, has_attachment

# Apply Preprocessing to 'body'
mail_data['Clean_Body'], mail_data['Num_URLs'], mail_data['Has_Attachment'] = zip(*mail_data['Body'].map(preprocess_email))

# Display Processed Data
print(mail_data.head())

#Feature Extraction
# TF-IDF Vectorization for Text Body
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=5,
    max_features=3000,
    lowercase=True
)

X_text = vectorizer.fit_transform(mail_data['Clean_Body'])

# Combine Text and Numeric Features
X_combined = sp.hstack([
    X_text,
    np.array(mail_data['Num_URLs']).reshape(-1,1),
    np.array(mail_data['Has_Attachment']).reshape(-1,1)
])

# Define Labels
y = mail_data['Label']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression Model
log_reg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg_model.fit(X_train, y_train)

# prediction on training data

prediction_on_training_data = log_reg_model.predict(X_train)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# Evaluate Model
y_pred = log_reg_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""Using Random forest Model"""

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

"""Using Naive Bayes Model"""

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# Naive Bayes Evaluation
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

#Predict on a Sample Email (for each model)
def predict_email(email_text, model):
    text, num_urls, has_attachment = preprocess_email(email_text)
    text_vector = vectorizer.transform([text])
    features = sp.hstack([
        text_vector,
        np.array([num_urls]).reshape(1, -1),
        np.array([has_attachment]).reshape(1, -1)
    ])
    prediction = model.predict(features)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example Prediction for each model
sample_email = """
Subject: Free Gift Card Offer!
Click here: http://spamlink.com to claim your reward.
"""

# Test on Logistic Regression Model
print("\nLogistic Regression Sample Email Prediction:", predict_email(sample_email, log_reg_model))

# Test on Random Forest Model
print("\nRandom Forest Sample Email Prediction:", predict_email(sample_email, rf_model))

# Test on Naive Bayes Model
print("\nNaive Bayes Sample Email Prediction:", predict_email(sample_email, nb_model))

# Example Prediction for each model
sample_email = """
Subject: re : new color printer  sorry ,  don ' t we need to know the cost , as well .  - - - - - - - - - - - - - - - - - - - - - - forwarded by kevin g moore / hou / ect on 12 / 14 / 99 08 : 15  am - - - - - - - - - - - - - - - - - - - - - - - - - - -  kevin g moore  12 / 14 / 99 08 : 09 am  to : shirley crenshaw / hou / ect @ ect , mike a roberts / hou / ect @ ect  cc :  subject : re : new color printer  this information was also sent to it purchasing .  i need to know what options we have and how soon it  can be delivered .  don ' t we need to know as well ? before purchase .  i also need a central location for this printer .  thanks  kevin moore  sam mentioned hp 4500 , i will check into it .  - - - - - - - - - - - - - - - - - - - - - - forwarded by kevin g moore / hou / ect on 12 / 14 / 99 08 : 05  am - - - - - - - - - - - - - - - - - - - - - - - - - - -  shirley crenshaw  12 / 14 / 99 07 : 55 am  to : kevin g moore / hou / ect @ ect  cc :  subject : re : new color printer  kevin :  what kind of information do you need ? i thought you were going to look  at some colored printer literature . sam seemed to be aware of a  colored printer that might work for us . ask him . i don ' t think we need  anything as big as " sapphire " .  it will be located in your area on the 19 th floor .  thanks !  kevin g moore  12 / 14 / 99 06 : 27 am  to : shirley crenshaw / hou / ect @ ect , vince j kaminski / hou / ect @ ect , mike a  roberts / hou / ect @ ect  cc :  subject : new color printer  we are in need of a new color printer .  we are also in the process of moving to the 19 th floor .  we need the color printer a . s . a . p .  if you would please , i need information concerning this  matter whereby , we can get the printer ordered and delivered  to our new location .  thanks  kevin moore
"""

# Test on Logistic Regression Model
print("\nLogistic Regression Sample Email Prediction:", predict_email(sample_email, log_reg_model))

# Test on Random Forest Model
print("\nRandom Forest Sample Email Prediction:", predict_email(sample_email, rf_model))

# Test on Naive Bayes Model
print("\nNaive Bayes Sample Email Prediction:", predict_email(sample_email, nb_model))