# Implementation of Chatbot using NLP

## AICTE Project of AI/ML

This project demonstrates a simple implementation of a chatbot using **Natural Language Processing (NLP)** techniques. The chatbot is built using **TF-IDF Vectorization**, **Logistic Regression**, and **Cosine Similarity** to classify user intents and generate appropriate responses. It is designed to run in a **Google Colab environment** and can be integrated with **Streamlit** for an interactive UI.

---

## 📖 Table of Contents

1. [Installation](#installation)
2. [Setup and Configuration](#setup-and-configuration)
3. [Chatbot Intents](#chatbot-intents)
4. [Training and Model Implementation](#training-and-model-implementation)
5. [Overall Flow](#overall-flow)
6. [Usage](#usage)

---

## 🛠 Installation

To install the necessary modules, run the following command in your Colab notebook:

```bash
!pip install nltk scikit-learn streamlit
```

### Installed Libraries:

- **NLTK**: A toolkit for natural language processing.
- **scikit-learn**: A machine learning library used for text vectorization and classification.
- **Streamlit**: A framework for building interactive web applications.

---

## ⚙️ Setup and Configuration

### 1️⃣ Importing Libraries

Import the required Python libraries:

```python
import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

### 2️⃣ Explanation of Imported Libraries

#### **NLTK (Natural Language Toolkit)**

A powerful NLP library that provides tools for:

- Text tokenization
- Stemming and lemmatization
- Tagging and parsing
- Sentiment analysis

#### **random**

A built-in Python module for generating random numbers and performing stochastic operations such as:

- Randomly selecting responses from a list
- Shuffling data

#### **os**

A standard Python library for interacting with the operating system:

- File and directory management
- Path manipulations
- Accessing environment variables

#### **ssl**

Handles **Secure Sockets Layer (SSL)** protocols, helping to manage secure network connections. Used to:

- Override the default HTTPS context
- Bypass SSL certificate verification (helpful in development environments)

#### **streamlit**

A framework for building interactive web applications and dashboards using Python, simplifying:

- Machine learning model integration
- Data visualizations
- Interactive chatbot UIs

#### **TfidfVectorizer (from sklearn.feature\_extraction.text)**

Converts text into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**, which helps:

- Identify important words in a document relative to a dataset
- Improve text-based machine learning performance

#### **LogisticRegression (from sklearn.linear\_model)**

A classification algorithm used to:

- Predict user intent categories
- Train the chatbot to classify user input accurately

---

## 🔒 SSL Configuration

Override Python’s default HTTPS context to bypass SSL certificate verification. This prevents SSL errors when downloading resources:

```python
ssl._create_default_https_context = ssl._create_unverified_context
```

---

## 📥 NLTK Data Setup

NLTK requires specific datasets for tokenization. This step ensures that the necessary resources are available:

```python
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')
```

- **punkt**: A tokenizer used to split text into words and sentences.

---

## 💬 Chatbot Intents

The chatbot uses predefined **intents** to understand and respond to user queries. Each intent is stored as a dictionary with:

- **tag**: A unique identifier for the intent.
- **patterns**: Sample user phrases triggering this intent.
- **responses**: Possible chatbot replies.

### Example Intent:

```json
{
    "tag": "greeting",
    "patterns": ["Hi", "Hello", "Hey"],
    "responses": ["Hello there!", "Hi!", "Greetings!"]
}
```

---

## 🎯 Training and Model Implementation

### 1️⃣ Preprocessing Data

Extracts training data from predefined intents:

```python
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
```
- random_state=0 : ensures reproducibility by making the training behavior consistent.
- max_iter=10000 : prevents premature stopping, ensuring proper model convergence.


```python
# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
```

### 2️⃣ Training the Model

Trains the chatbot using **TF-IDF Vectorization** and **Logistic Regression**:

```python
# Convert text data into numerical format
x = vectorizer.fit_transform(patterns)
y = tags

# Train the classifier
clf.fit(x, y)
```

### 3️⃣ Evaluating Accuracy

Measures how well the chatbot classifies user inputs:

```python
from sklearn.metrics import accuracy_score

y_pred = clf.predict(x)
print("Training Accuracy:", accuracy_score(y, y_pred))
```

🔹 **Expected Output:** Training Accuracy: ~0.95 (95%)

---

## 🔄 Overall Flow

1. **User Input**: The chatbot receives a message.
2. **TF-IDF Vectorization**: Converts text into numerical form.
3. **Intent Prediction**: Uses Logistic Regression and Cosine Similarity to identify the best match.
4. **Response Selection**: Returns a random response from the matched intent.

---

## 🤖 Chatbot Function Implementation
```python
from sklearn.metrics.pairwise import cosine_similarity

# Python function to chat with the chatbot
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])

    # Use both ML classification and similarity matching
    predicted_tag = clf.predict(input_text_vectorized)[0]

    # Cosine Similarity for better matching
    similarities = cosine_similarity(input_text_vectorized, x)
    best_match_index = similarities.argmax()
    best_match_tag = y[best_match_index]

    # Choose the more reliable result
    final_tag = best_match_tag

    for intent in intents:
        if intent['tag'] == final_tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."
```

### 📌 Summary of Chatbot Function Implementation

The `chatbot(input_text)` function processes user input and generates an appropriate response using **Machine Learning (ML) classification** and **Cosine Similarity**.

#### **🔍 How It Works:**
1. **Vectorization of Input**:  
   - The user’s input is converted into a numerical form using **TF-IDF Vectorization**.

2. **Machine Learning Prediction**:  
   - The **Logistic Regression** model predicts the most likely **intent tag** based on trained data.

3. **Cosine Similarity Matching**:  
   - The function calculates the **cosine similarity** between the input text and all training patterns.
   - It finds the most similar pattern and retrieves its corresponding **intent tag**.

4. **Selecting the Best Match**:  
   - Instead of relying solely on ML classification, the chatbot chooses the **most similar intent** from the dataset for better accuracy.

5. **Generating a Response**:  
   - Once the final **intent tag** is determined, the chatbot randomly selects a response from the predefined **intent responses**.

6. **Fallback Response**:  
   - If no match is found, the chatbot defaults to **"Sorry, I didn't understand that."**

✅ **Why Use Both ML and Cosine Similarity?**  
Using **both techniques** improves accuracy and ensures that even if the ML model misclassifies an input, the chatbot can still match it based on text similarity.

---

## ▶️ Usage

### Process 1: 
1. Run the chatbot in Google Colab or a local Python environment.
2. Enter a message in the input field.
3. The chatbot will process the input and return a relevant response.

```python

user_input = "How old are you?"
response = chatbot(user_input)
print(response)
     
```
🔹Age is just a number for me.

### Process 2:
1. Only run the function
2. Interact with chatbot infintely ules  you want to stop

```python
#making a continuous messaging function
greeting_message=random.choice(intents[0]['responses'][3:])
goodbye_patterns=intents[1]['patterns']
user_input=input(f"{greeting_message}\n")
print(chatbot(user_input))
while(True):
  user_input_message=input()
  if(user_input_message in goodbye_patterns):
    print(chatbot(user_input_message))
    break
  else:
    response=chatbot(user_input_message)
    print(response)
```
🔹 I need help  
🔹 Sure, I'm here to help. What do you need assistance with?  
🔹 What's the weather like?  
🔹 I don't have current weather data, but you can check a weather website or app for up-to-date info.  
🔹 Tell me a joke  
🔹 Why don't scientists trust atoms? Because they make up everything!  
🔹 See ya  
🔹 Bye! Have a great day!  



🚀 **Future Improvements:**

- Expand training data for better accuracy.
- Use **Word Embeddings (spaCy, FastText)** for enhanced NLP.
- Implement **Deep Learning (LSTMs, Transformers)** for smarter interactions.

🎯 **Happy Chatbot Building!**

