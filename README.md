# Spam Mail Prediction

A project leveraging machine learning to classify emails as spam or ham. This project explores the use of Logistic Regression, Random Forest, and Naive Bayes models for effective spam detection.  

## Team Members  

- Esraa Matarawy Abdelmoniem
- Roqaya Hassan Hamed
- Tasnim Khaled Mohamed 


## Table of Contents  

1. [Google Colab Code](Google_Colab_Code)  
2. [HuggingFace Files](HF)  
3. [Project Report](Spam_Mail_Prediction.pdf)  
4. [Demo](22.12.2024_23.04.52_REC.mp4)  

## Introduction  

Spam emails pose a persistent challenge in cybersecurity. This project employs NLP techniques and machine learning models to detect and classify spam emails effectively, focusing on improving accuracy, reducing false positives, and handling class imbalances.  

## Dataset  

- **Source**: [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data)  
- **Size**: 5,695 unique email records after cleaning.  
- **Features**: Email body text, presence of URLs, and attachments.  

## Methodology  

### Preprocessing  
- Text cleaning: Lowercasing, removing URLs, and non-alphanumeric characters.  
- Feature engineering: TF-IDF vectorization, URL count, and attachment indicators.  

### Models  
1. Logistic Regression  
2. Random Forest  
3. Naive Bayes  

### Metrics  
- Accuracy  
- Precision  
- Recall  
- F1-score  

## Results  

| Model               | Accuracy | Precision | Recall | F1-Score |  
|---------------------|----------|-----------|--------|----------|  
| Logistic Regression | 98.95%   | High      | High   | High     |  
| Random Forest       | 98.95%   | High      | High   | High     |  
| Naive Bayes         | 98.43%   | Moderate  | High   | Moderate |  

## Real-World Applications  

- **Personal Email Security**: Protect individuals from phishing and malicious content.  
- **Enterprise Systems**: Enhance cybersecurity by filtering emails and reducing phishing risks.  

## Future Work  

- Explore deep learning models like RNNs or Transformers for improved accuracy.  
- Integrate metadata such as sender reputation and email domain analysis.  
- Develop real-time filtering systems for deployment.  

## Setup and Execution  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/Spam_Mail_Prediction_ML.git  
   cd Spam_Mail_Prediction_ML 
   ```  
2. Install the required Python libraries:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the notebook on Google Colab for execution.  

## Links  

- **Google Colab**: [Run on Colab]([https://colab.research.google.com/your-link-here](https://colab.research.google.com/drive/1tgQtrE2ngGZixsrlNi2vqijYwoJ_EvL-?usp=sharing#scrollTo=XSkYEvNiISCT))  
- **Hugging Face**: [Project Repository]([https://huggingface.co/your-link-here](https://huggingface.co/spaces/roqayahassan/MLProject))  
