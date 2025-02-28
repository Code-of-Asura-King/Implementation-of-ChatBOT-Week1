# Implementation-of-ChatBOT using NLP
AICTE project of AI/ML


This project demonstrates a simple implementation of a chatbot using Natural Language Processing (NLP) techniques. The project is designed to run in a Google Colab environment and leverages popular libraries such as NLTK, scikit-learn, and Streamlit.

## Table of Contents

- [Installation](#installation)
- [Setup and Configuration](#setup-and-configuration)
- [Chatbot Intents](#chatbot-intents)
- [Overall Flow](#overall-flow)
- [Usage](#usage)

## Installation

To install the necessary modules, run the following command in your Colab notebook:

```bash
!pip install nltk scikit-learn streamlit
```
### This will install:

- NLTK: A toolkit for natural language processing.
- scikit-learn: A machine learning library used for text vectorization and classification.
- Streamlit: A framework for building interactive web applications.


## Setup and Configuration
Importing Libraries
Import the required Python libraries:

```bash
import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```
# Explanation of Imported Libraries

Below is an overview of each library and module imported in the code:

- **nltk (Natural Language Toolkit):**  
  A comprehensive library for natural language processing (NLP) tasks. It provides tools for:
  - Text tokenization
  - Stemming and lemmatization
  - Tagging and parsing
  - And much more  
  This makes it a popular choice for developing NLP applications.

- **random:**  
  A built-in Python module that provides functions for generating random numbers and performing random operations. It is useful for tasks such as:
  - Randomly selecting responses from a list
  - Shuffling data
  - Other stochastic processes in your application

- **os:**  
  A standard Python library for interacting with the operating system. It offers functionalities like:
  - File and directory management
  - Path manipulations
  - Accessing environment variables

- **ssl:**  
  This module handles Secure Sockets Layer (SSL) protocols and helps manage secure network connections. In this context, it is used to:
  - Override the default HTTPS context
  - Bypass SSL certificate verification when necessary (helpful in certain development environments)

- **streamlit:**  
  A framework for building interactive web applications and dashboards using Python. It simplifies the process of creating user interfaces for:
  - Machine learning models
  - Data visualizations
  - Other interactive applications

- **TfidfVectorizer (from sklearn.feature_extraction.text):**  
  A tool from the scikit-learn library that converts a collection of raw documents into a matrix of TF-IDF features. TF-IDF stands for:
  - **Term Frequency-Inverse Document Frequency**  
    It helps quantify the importance of words in a document relative to a corpus, which is key for many text-based machine learning tasks.

- **LogisticRegression (from sklearn.linear_model):**  
  A machine learning algorithm used for classification tasks. In scikit-learn, this model is employed to:
  - Predict categorical outcomes based on input features
  - Classify user input into different intent categories or other classes



## **SSL Configuration**
Override Python’s default HTTPS context to bypass SSL certificate verification. This is sometimes necessary when accessing resources over HTTPS from servers with self-signed or improperly configured certificates. This is helpful in avoiding issues during resource downloads:

```bash
ssl._create_default_https_context = ssl._create_unverified_context
```

## **NLTK Data Setup**
Append a local directory to NLTK’s data search path and download the punkt tokenizer, which is essential for tokenizing sentences and words:

```bash
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')
```
## **Chatbot Intents**
The chatbot uses predefined intents to understand and respond to user queries. Each intent is represented as a dictionary with three keys:

- **tag:** A unique identifier for the intent.
- **patterns:** Sample phrases that a user might say to trigger this intent.
- **responses:** Predefined replies that the chatbot can use when the intent is recognized.
