
# Your Gradio app code here
import gradio as gr
import joblib
import numpy as np
from scipy.sparse import hstack

# Load your model and vectorizer
model = joblib.load("spam_classifier_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict_spam(clean_body, num_urls, has_attachment):
    X_text = vectorizer.transform([clean_body])
    X_combined = hstack([
        X_text,
        np.array([num_urls]).reshape(-1, 1),
        np.array([has_attachment]).reshape(-1, 1)
    ])
    prediction = model.predict(X_combined)[0]
    return "Spam" if prediction == 1 else "Not Spam"

interface = gr.Interface(
    fn=predict_spam,
    inputs=[
        gr.Textbox(lines=5, label="Email Body"),
        gr.Slider(0, 50, step=1, label="Number of URLs"),
        gr.Radio([0, 1], label="Has Attachment (0 = No, 1 = Yes)")
    ],
    outputs=gr.Text(label="Prediction"),
    title="Spam Email Classifier",
    description="Classify emails as Spam or Not Spam."
)
interface.launch()
