import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# Updated dataset with loan_options intent
data = [
    {"text": "I need a loan", "intent": "loan_inquiry"},
    {"text": "I am looking for a loan", "intent": "loan_inquiry"},
    {"text": "What is the status of the loan", "intent": "loan_status"},
    {"text": "I wanted a loan", "intent": "loan_inquiry"},
    {"text": "I'm looking for a loan", "intent": "loan_inquiry"},
    {"text": "Can you help with a loan", "intent": "loan_inquiry"},
    {"text": "What are the loans available?", "intent": "loan_options"},
    {"text": "Can you show me the available loans?", "intent": "loan_options"},
    {"text": "I want to see all loan types", "intent": "loan_options"},
    {"text": "List all loan options", "intent": "loan_options"},
    {"text": "thanks for your response", "intent": "thanks"},
    {"text": "thank you for answering", "intent": "thanks"},
    {"text": "thanks a lot", "intent": "thanks"},
    {"text": "thank you so much", "intent": "thanks"},
    {"text": "ok done hmm", "intent": "acknowledgment"},
    {"text": "okay done got it", "intent": "acknowledgment"},
    {"text": "hmm got it", "intent": "acknowledgment"},
    {"text": "hi", "intent": "greeting"},
    {"text": "hello", "intent": "greeting"},
    {"text": "hey", "intent": "greeting"},
    {"text": "greetings", "intent": "greeting"},
    {"text": "hai", "intent": "greeting"},
    {"text": "What is your name?", "intent": "asking_name"},
    {"text": "Who are you?", "intent": "asking_name"},
    {"text": "Tell me your name", "intent": "asking_name"},
    {"text": "Who am I talking to?", "intent": "asking_name"},
    {"text": "Can I know your name?", "intent": "asking_name"},
    # ... add more samples for robustness
]

texts = [item['text'] for item in data]
intents = [item['intent'] for item in data]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(texts)
max_length = max([len(s.split()) for s in texts])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Convert intents to one-hot encoding
encoder = LabelEncoder()
encoder.fit(intents)
encoded_intents = encoder.transform(intents)
categorical_intents = to_categorical(encoded_intents)

# LSTM Model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(len(set(intents)), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(padded_docs, categorical_intents, epochs=50, verbose=0)

# Save the model
model.save("intent_model.h5")

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open('encoder.pkl', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
