from flask import Flask, render_template, request, jsonify
from keras.utils import pad_sequences

import loan_prediction_recommendation as lpr
import pandas as pd

from tensorflow.keras.models import load_model
import pickle
import numpy as np

import os
import re

from gtts import gTTS

app = Flask(__name__)

PROMPTS = [
    "Enter the Loan Amount required: ",
    "Enter the Term (1 for Short, 2 for Long): ",
    "Enter your Credit Score: ",
    "Enter your household income: ",
    "Enter the number of years of experience in your job (e.g. 1, 2, 3): ",
    "Enter 1 for 'Rent', 2 for 'Home Mortgage', and 3 for 'Own Home': ",
    "Enter 1 for 'Personal Loan', 2 for 'Home Loan', 3 for 'Business Loan', 4 for 'Education Loan', 5 for 'Vehicle Loan', 6 for 'shopping loan', and 7 for 'wedding loan': ",  # you can replace this with actual loan types
    "Enter your Monthly Debt: ",
    "Enter your Years of Credit History: "
]

RESPONSES = []
STATE = "INITIAL"

# Load the model, tokenizer, and encoder
model = load_model("intent_model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('encoder.pkl', 'rb') as handle:
    encoder = pickle.load(handle)

# Text-to-Speech function
def text_to_speech(text, language='en'):
    from gtts import gTTS
    # Convert text to speech
    tts = gTTS(text=text, lang=language, slow=False)
    # Generate a unique filename
    filename = os.path.join("static", "audio", f"{hash(text)}.mp3")
    tts.save(filename)
    return filename


def predict_intent(message):
    # Tokenize and pad the input message
    encoded_msg = tokenizer.texts_to_sequences([message])
    max_length = 7
    padded_msg = pad_sequences(encoded_msg, maxlen=max_length, padding='post')
    # Predict the intent
    intent_probabilities = model.predict(padded_msg)
    predicted_intent_index = np.argmax(intent_probabilities[0])
    predicted_intent = encoder.classes_[predicted_intent_index]

    return predicted_intent

@app.route('/chatbot_query', methods=['POST'])
def chatbot_query():
    global STATE
    user_query = request.json.get('query', '').lower()

    # Helper function to generate both text and audio responses
    def generate_response(text):
        audio_path = text_to_speech(text)
        return jsonify({'response': text, 'audio_path': audio_path})

    # If the user has indicated they are an existing customer
    if user_query == "1" and STATE == "INITIAL":
        RESPONSES.clear()
        STATE = "EXISTING_CUSTOMER"
        return generate_response("Please Enter Your Customer Code.")

    # If the user has indicated they are a new customer
    elif user_query == "2" and STATE == "INITIAL":
        STATE = "NEW_CUSTOMER"
        return generate_response(PROMPTS[0])

    # If the user is entering their customer code
    elif STATE == "EXISTING_CUSTOMER":
        RESPONSES.append(user_query)
        STATE = "AWAIT_PREDICTION"
        return generate_response("Please wait for a minute. I will check your profile and come back.")

    # If the system is processing the customer code
    elif STATE == "AWAIT_PREDICTION":
        cust_code = int(RESPONSES[-1])
        result = lpr.existing_customer_code(cust_code)
        STATE = "INITIAL"
        RESPONSES.clear()
        return generate_response(result)

    # If the user is entering details as a new customer
    elif STATE == "NEW_CUSTOMER":
        RESPONSES.append(user_query)
        if len(RESPONSES) == len(PROMPTS):
            STATE = "PROCESSING"
            return generate_response("Please wait for a minute. I will check your loan eligibility and come back.")
        else:
            return generate_response(PROMPTS[len(RESPONSES)])


    elif STATE == "PROCESSING":

        term_mapping = {1: 'Short Term', 2: 'Long Term'}

        home_ownership_mapping = {1: 'Rent', 2: 'Home Mortgage', 3: 'Own Home'}

        loan_type_mapping = {

            1: 'Personal Loan',

            2: 'Home Loan',

            3: 'Business Loan',

            4: 'Education Loan',

            5: 'Vehicle Loan',

            6: 'shopping loan',

            7: 'wedding loan'

        }

        sample_data = {

            'Current Loan Amount': [int(RESPONSES[0])],

            'Term': [term_mapping[int(RESPONSES[1])]],

            'Credit Score': [int(RESPONSES[2])],

            'household_income': [int(RESPONSES[3])],

            'Years in current job': [int(RESPONSES[4])],

            'Home Ownership': [home_ownership_mapping[int(RESPONSES[5])]],

            'Loan_type': [loan_type_mapping[int(RESPONSES[6])]],

            'Monthly Debt': [int(RESPONSES[7])],

            'Years of Credit History': [int(RESPONSES[8])]

        }

        sample_data_df = pd.DataFrame(sample_data)

        eligibility, recommended_loans = lpr.new_customer_code(sample_data_df)

        if eligibility == "Eligible":

            result = f"Hey, you are {eligibility} for the loan and also, we recommend the following loans: {', '.join(recommended_loans)}. Additionally, these are pre-approved loans for you."

        else:

            result = f"Sorry, you are '{eligibility}' for this loan. However, based on your profile, we recommend the following loans: {', '.join(recommended_loans)}."

        RESPONSES.clear()  # Clear the list after processing

        STATE = "INITIAL"  # Reset the state after processing

        return generate_response(result)

        # If we're in the INITIAL state or none of the above conditions are met
    if STATE == "INITIAL":
        predicted_intent = predict_intent(user_query)

        greeting_pattern = re.compile(r'^(hi|hello|hey)\s*(cassie)?\s*$', re.IGNORECASE)
        acknowledgment_pattern = re.compile(r'^(thanks|welcome|thank you|ok|okay|Bye)\s*$', re.IGNORECASE)

        if greeting_pattern.match(user_query):
            return generate_response("Hello, How can I Assist you?")

        elif acknowledgment_pattern.match(user_query):
            return generate_response('I am happy to assist you.')

        elif predicted_intent == "thanks":
            return generate_response("You're welcome! It's my pleasure to assist you.")

        elif predicted_intent == "greeting":
            return generate_response("Sorry, I can assist you only in the Loan Application.")

        elif predicted_intent == "loan_inquiry":
            return generate_response("Are you an Existing Customer (Enter 1) or New Customer (Enter 2)?")

        elif predicted_intent == "loan_options":
            loans = [
                'Business Loan', 'Education Loan', 'Home Loan',
                'Personal Loan', 'shopping loan', 'Vehicle Loan', 'wedding loan'
            ]
            formatted_loans = "\n".join([f"{idx + 1}. {loan}" for idx, loan in enumerate(loans)])
            return generate_response(formatted_loans)

        elif predicted_intent == "acknowledgment":
            return generate_response('I am happy to assist you.')

        elif predicted_intent == "asking_name":
            return generate_response('You can call me Cassie. How can I assist you further?')

        elif predicted_intent == "irrelevant_input":
            return generate_response('Sorry, I can assist you only related to loan details.')


    return jsonify({'response': "Sorry, I can support you only in the Loan Application."})

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        if request.form.get('customer_type') == 'existing':
            cust_code = int(request.form.get('cust_code'))
            result = lpr.existing_customer_code(cust_code)
        elif request.form.get('customer_type') == 'new':
            term_mapping = {1: 'Short Term', 2: 'Long Term'}
            home_ownership_mapping = {1: 'Rent', 2: 'Home Mortgage', 3: 'Own Home'}
            loan_type_mapping = {
                1: 'Personal Loan',
                2: 'Home Loan',
                3: 'Business Loan',
                4: 'Education Loan',
                5: 'Vehicle Loan',
                6: 'shopping loan',
                7: 'wedding loan'
            }

            sample_data = {
                'Current Loan Amount': [int(request.form.get('loan_amount'))],
                'Term': [term_mapping[int(request.form.get('term'))]],
                'Credit Score': [int(request.form.get('credit_score'))],
                'household_income': [int(request.form.get('income'))],
                'Years in current job': [int(request.form.get('job_experience'))],
                'Home Ownership': [home_ownership_mapping[int(request.form.get('home_ownership'))]],
                'Loan_type': [loan_type_mapping[int(request.form.get('loan_type'))]],
                'Monthly Debt': [int(request.form.get('monthly_debt'))],
                'Years of Credit History': [int(request.form.get('credit_history'))]
            }
            sample_data_df = pd.DataFrame(sample_data)

            # Use the predict_loan_status function and ensure it returns the desired output
            eligibility, recommended_loans = lpr.new_customer_code(sample_data_df)

            # Here you format the result string based on the eligibility and recommended loans
            if eligibility == "Eligible":
                result = f"Hey, you are {eligibility} for the loan and also, we recommend the following loans: {', '.join(recommended_loans)}. Additionally, these are pre-approved loans for you."
            else:
                result = f"Sorry, you are '{eligibility}' for this loan. However, based on your profile, we recommend the following loans: {', '.join(recommended_loans)}."

    return render_template('index.html', result=result)



@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/single')
def single():
    return render_template('single.html')
if __name__ == "__main__":
    app.run(debug=True)
