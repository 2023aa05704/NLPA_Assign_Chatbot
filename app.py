# # # GUS: Simple Frame-based Dialogue System Integrated with a Chatbot
# # # Objective:
# # # Develop a frame-based dialogue system (GUS) for handling user queries and facilitating interactions through a chatbot.
# # # The system should use a frame-based approach to represent information and manage dialogue.
# # # It should be capable of handling user requests, filling in relevant slots based on predefined templates, and providing a coherent conversation flow.

# Libraries and data imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
import logging
from nltk.corpus import stopwords
import nltk
import spacy
import ast

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask App for chatbot UI
app = Flask(__name__)


# # # Data Collection and Preprocessing: (3 Marks)
# # # a) Data Collection:
# # # i. Select a domain for your frame-based dialogue system ( customer service).
# # # Collect a relevant dataset that contains sample dialogues, question-answer pairs, or predefined slot values.
# # #  This dataset may include conversation logs, FAQs, or structured data about the domain.
# # #  ii) Preprocess the data to ensure it is ready for dialogue management.
# # # This may include tokenization, slot identification,
# # # and preparing structured data representations (frames) that correspond to possible user inputs.

# This application uses intent, slot based dataset (chatbot_data_7k.csv) for a order processing customer service bot.
# It is a csv dataset with about 7000 records consisting of 7 intents, each having 1000 records

# step 1: loading and cleaning data
def preprocess_data(file_path):
    """Load and preprocess dataset."""
    try:
        data = pd.read_csv(file_path)
        data['clean'] = (data['instruction'] + " " + data['intent'] + " " + data['response']).apply(lambda x: ' '.join([word for word in str(x).lower().split() if word not in stop_words]))
        return data
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

data = preprocess_data('./data/chatbot_data_7k.csv')
logging.info("Dataset loaded and preprocessed.")

# # # b) Frame-based Model Development:
# # # i. Implement a frame-based dialogue system using a simple template or slot-filling approach.
# # # Frames should represent the structure of the conversation, where each frame corresponds to a specific request and its associated slots (e.g., "flight booking" with slots like "date," "destination," etc.).
#  # # ii. Ensure that the dialogue system can process user inputs, identify relevant slots, and fill those slots with the correct information based on the predefined templates.

# A frame with intent, slots & responses to help facilitate dialogue with the users.
frame = [
            {
                "intent": "create_order",
                "slots": {
                    "quantity": None,
                    "product_name": None,
                },
                "responses": {
                    "ask_quantity": "Can you provide the quantity that you want to order?",
                    "ask_product_name": "What item do you want to order?",
                    "confirmation": "I understand that you want to order {quantity} units of {product_name}. Let me assist you. \n Is there anythign else I can help you with?"
                }
            },
            {
                "intent": "cancel_order",
                "slots": {
                    "order_number": None,
                },
                "responses": {
                    "ask_order_number": "Can you provide the order number that you want to cancel?",
                    "confirmation": "I understand that you want to cancel the order number {order_number}. I'll initate and confirm once the order is cancelled. \n Is there anythign else I can help you with?"
                }
            },
            {
                "intent": "check_order_status",
                "slots": {
                    "order_number": None,
                },
                "responses": {
                    "ask_order_number": "Can you provide the order number that you want to check?",
                    "confirmation": "I understand that you want to check the status of the order number {order_number}. I'll fetch and share the status. \n Is there anythign else I can help you with?"
                }
            },
            {
                "intent": "contact_customer_support",
                "slots": {
                },
                "responses": {
                    "confirmation": "To contact customer support, please provide your inquiry and we will direct you to the right team to assist you. \n Is there anythign else I can help you with?"
                }
            },
            {
                "intent": "request_refund",
                "slots": {
                    "order_number": None,
                },
                "responses": {
                    "ask_order_number": "Can you provide the order number for us to initiate refund?",
                    "confirmation": "I have created the refund request for order number {order_number}. \n Is there anythign else I can help you with?"
                }
            },
            {
                "intent": "track_refund",
                "slots": {
                    "order_number": None,
                },
                "responses": {
                    "ask_order_number": "Can you provide the order number for us to check refund status?",
                    "confirmation": "I'll check refund for {{order_number}} and share its status with you.. \n Is there anythign else I can help you with?"
                }
            }
]

# Model training to correctly identify intent based on user input
# The preprocessed dataset consists of intruction, category and response text which provides the context of the action and the intent column is the associated class we want to predict.

# Train Word2Vec model for vectorization of the context

logging.info("Training Word2Vec model.")
all_sentences = data['clean'].apply(str.split).tolist()
word2vec_model = Word2Vec(sentences=all_sentences, vector_size=300, window=5, min_count=1, workers=4)

# Convert sentence to weighted vector so that it can be mapped to a features & labels
def sentence_to_vector(sentence, model=word2vec_model):
    """Convert a sentence to a weighted vector by averaging word embeddings."""
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model.wv:
            # Weight the vector by the word frequency (inverse document frequency)
            weight = 1.0 / (1.0 + model.wv.get_vecattr(word, 'count'))
            word_vectors.append(model.wv[word] * weight)
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

logging.info("Preparing features and labels.")
data['vector'] = data['clean'].apply(lambda x: sentence_to_vector(x, word2vec_model))
X = np.stack(data['vector'].values)
y = data['intent']

# Spliting dataset for training of a multiclass classification model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForestClassifier
def train_classifier(X_train, y_train):
    """Train a RandomForest classifier."""
    clf = RandomForestClassifier(n_estimators=400, random_state=1)
    clf.fit(X_train, y_train)
    return clf

classifier = train_classifier(X_train, y_train)
logging.info("Classifier trained.")

# This classifier will help us find the intent close to the input based on the class


# # # Frame-based Dialogue System: (3 Marks)
# # # a) Query Processing:
# # # i. Develop Natural Language Processing (NLP) techniques to process user inputs and extract the relevant slots from the dialogue.
#           This can include named entity recognition (NER), part-of-speech tagging, and syntactic parsing.
# # # ii. Implement query interpretation strategies to understand the user's intent and identify the relevant frame and slots to be filled.
#           This might involve extracting keywords and matching them to specific slots in the predefined templates.

# Predict Intent function to find the class (intent) based on user input
def predict_intent(user_input):
    """Predict intent from user input with scaled embedding."""
    user_embedding = sentence_to_vector(user_input)
    user_embedding_scaled = scaler.transform([user_embedding])  # Scale the embedding
    predicted_intent = classifier.predict(user_embedding_scaled)[0]
    logging.info(f"Predicted intent: {predicted_intent}")
    return predicted_intent

# Load spaCy's pre-trained model - This will help to the NER for matching the user input with the slot values
nlp = spacy.load("en_core_web_lg")

# extract_slots function will help extract slots for a given intent and response using NER and pattern matching
def extract_slots(user_response, intent):
    # Find the intent frame from the frame based on the intent
    intent_frame = {}
    intent_frame = next((dict(f) for f in frame if f['intent'] == intent), None)
    logging.debug(f'intent frame identified based on the intent {intent} is {intent_frame} and all the slots should be None')
    if not intent_frame:
        return "Intent not found."

    slots = intent_frame['slots']
    logging.debug(f"slots from the intent frame: {slots}")
    if not slots:
        return None

    # Process the user response with spaCy for NER
    doc = nlp(user_response)

    # Slot extraction logic
    extracted_slots = {}
    for slot in slots:
        if slot == "quantity":
            # Extract numbers for quantity
            for token in doc:
                if token.like_num:
                    extracted_slots[slot] = token.text
                    break
        elif slot == "order_number":
            # Extract patterns resembling large order numbers
            for token in doc:
                if token.text.isnumeric() and len(token.text) >= 5:
                    extracted_slots[slot] = token.text
                    break
        elif slot == "product_name":
            # Extract product name using Named Entities
            for ent in doc.ents:
                logging.debug(f"Entity: {ent}")
                if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
                    extracted_slots[slot] = ent.text
                    break

    logging.info(f'slots extracted: {extracted_slots}')

    # Fill in extracted values into the frame
    for key, value in extracted_slots.items():
        intent_frame['slots'][key] = value

    return intent_frame


# # # b) Dialogue Management:
# # # i. Implement a dialogue manager that keeps track of conversation context and fills in the slots with information extracted from the user’s inputs.
# # # The manager should be able to handle multi-turn conversations by maintaining context across turns and prompting users for missing information.
# # # ii. Ensure that the dialogue manager uses the frame-based approach to manage the conversation flow, dynamically updating the frame as the conversation progresses.
# # # The system should handle various dialogue patterns, including clarification, confirmation, and error recovery.


# Below function will help process the user input
# 1. find the best matching intent based on Word2Vec + RandomForestClassier (above function call)
# 2. find the slot based on intent and extract the slot value using NER and pattern match (above function call)
# 3. It will also help take action based on session - if it is a continuation of chat or a new chat

def fill_slots_and_respond(user_input, session):
    if session.lower() == "none" or not session or session == None: # new chat session
        intent_frame = {}
        intent = predict_intent(user_input)
        intent_frame = extract_slots(user_input, intent).copy()
    else:                                                           # existing session
        intent_frame = ast.literal_eval(session)
        intent = intent_frame['intent']
        new_intent_frame = extract_slots(user_input, intent).copy()
        intent_frame['slots'] = new_intent_frame['slots']

    # checking if there's any slot missing value
    if intent_frame and 'slots' in intent_frame:
        for slot, value in intent_frame['slots'].items():
            logging.debug(f'slot and value are {slot} and {value}')
            if value is None:
                return intent_frame['responses'][f"ask_{slot}"], intent_frame.copy()

    # final response when all the slots are fileld
    return intent_frame['responses']['confirmation'].format(**intent_frame['slots']), "None"

# # # Chatbot Integration for Interactive Dialogue: (2 Marks)
# # # a) Conversational Interface:
# # # i. Integrate a chatbot framework (e.g., Rasa, Dialogflow, or Botpress) to provide an interactive interface where users can ask questions or make requests. The chatbot should use the frame-based dialogue system to understand the user’s queries and respond appropriately.
# # # ii. Ensure that the chatbot provides accurate and contextually relevant responses while maintaining a natural flow of conversation.

# This application using Flask framework for the UI management.
# Using simple HTML and CSS, it is able to provide a chatbot interface

@app.route('/')
def home():
    return render_template('index.html')

# # # b) Dialogue Flow Management:
# # # i. Implement a system to manage dialogue turns, ensuring that the chatbot can handle interruptions, clarification requests, and multi-turn interactions. The system should track what information has been provided and what is still missing.
# # # ii. Ensure that the chatbot can handle follow-up questions or changes in the conversation topic, dynamically adjusting the frame as needed.

# Below function helps manage the user response and session using cookies
# It manages the state of slot values to decide on new chat vs continuation

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    session = request.cookies.get('session', default="None")
    logging.debug(f'user response: {user_input} and session: {session}')
    response_text, session = fill_slots_and_respond(user_input, session)
    response = make_response(jsonify({"response": response_text}))
    response.set_cookie('session', str(session), max_age=60)
    return response

if __name__ == '__main__':
    app.run(debug=True)