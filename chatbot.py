import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# SSL configuration to bypass SSL certification verification
ssl._create_default_https_context = ssl._create_unverified_context

# Specifying datasets for tokenization
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# Uploading intent file for chat
file_path = os.path.abspath("C:/Users/Anji/Desktop/New folder/intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Preprocessing data for training
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=100000)
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])

    # Use ML classification
    predicted_tag = clf.predict(input_text_vectorized)[0]

    # Cosine Similarity for better matching
    similarities = cosine_similarity(input_text_vectorized, x)
    best_match_index = similarities.argmax()
    best_match_tag = y[best_match_index]

    # Choose the best matching tag
    final_tag = best_match_tag

    for intent in intents:
        if intent['tag'] == final_tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store conversation history
if "chat_running" not in st.session_state:
    st.session_state.chat_running = True  # Control conversation loop
if "show_restart" not in st.session_state:
    st.session_state.show_restart = False  # Control visibility of restart button
if "restart" not in st.session_state:
    st.session_state.restart = False  # Control the rerun process

def restart_chat():
    st.session_state.messages = []
    st.session_state.chat_running = True
    st.session_state.show_restart = False
    st.session_state.restart = True  # Set flag to restart

def main():
    st.title("Intents-Based Chatbot using NLP")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if st.session_state.restart:
        st.session_state.restart = False  # Reset flag
        st.rerun()  # Restart the Streamlit app

    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message below to start chatting.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        greeting_message = random.choice([resp for intent in intents if intent['tag'] == 'greeting' for resp in intent['responses'][2:]])
        goodbye_patterns = [pattern.lower() for intent in intents if intent['tag'] == 'goodbye' for pattern in intent['patterns']]

        if not st.session_state.messages:
            st.session_state.messages.append(("Chatbot", greeting_message))

        for idx, (role, msg) in enumerate(st.session_state.messages):
            st.text_area(role, value=msg, height=70, key=f"{role}_{idx}", disabled=True)

        if st.session_state.chat_running:
            user_input = st.text_input("You:", key=f"user_input_{len(st.session_state.messages)}")
            if user_input:
                response = chatbot(user_input)

                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("Chatbot", response))

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])

                if user_input.lower() in goodbye_patterns:
                    st.session_state.chat_running = False
                    st.session_state.messages.append(("Chatbot", "Thank you for chatting with me. Have a great day!"))
                    st.session_state.show_restart = True

                st.rerun()

        if st.session_state.show_restart:
            st.button("Start New Chat", on_click=restart_chat)

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.write("## About the Project")
        st.write("This project is designed to build an intelligent chatbot that understands user intent and provides meaningful responses using Natural Language Processing (NLP). Built with Streamlit, it offers an interactive user-friendly interface.")
        
        st.subheader("Project Features")
        st.write("- Uses NLP and machine learning (Logistic Regression) for intent classification.")
        st.write("- Supports conversation logging for improved user interaction analysis.")
        st.write("- Provides a seamless chat experience with automated responses.")
        st.write("- Allows restarting the chat session for new interactions.")
        
        st.subheader("Future Enhancements")
        st.write("- Expand dataset to support a wider range of topics and intents.")
        st.write("- Integrate advanced deep learning models like transformers for better understanding.")
        st.write("- Implement speech-to-text functionality for voice-based interactions.")
        st.write("- Enhance UI/UX with richer chatbot responses, including multimedia integration.")

if __name__ == '__main__':
    main()
