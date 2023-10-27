import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def existing_customer_code(cust_code):
    df = pd.read_excel("Merged_Loan&Recommender_data.xlsx")
    return get_recommendations(cust_code, df)

def get_recommendations(cust_code, df):
    # Extracting data for the provided cust_code
    user_data = df[df['cust_code'] == cust_code]

    # Initialize recommendations list
    recommendations = []
    response = []

    if not user_data.empty:
        loan_status = user_data['Loan Status'].iloc[0]
        response.append(f"Thanks for your patience. You, with customer code {cust_code}, have already taken the following loans previously:")

        # Check loans taken by user and their status
        for loan in ['Business Loan', 'Education Loan', 'Home Loan', 'Personal Loan', 'shopping loan', 'Vehicle Loan',
                     'wedding loan']:
            if user_data[loan].iloc[0] == 1:
                response.append(f"- {loan}: {loan_status}")

        if user_data['Home Loan'].iloc[0] == 1 and loan_status == 'Fully Paid':
            recommendations.extend(['Vehicle Loan', 'Business Loan'])
        if user_data['Vehicle Loan'].iloc[0] == 1:
            if loan_status == 'Fully Paid':
                recommendations.extend(['Home Loan', 'Personal Loan'])
            elif loan_status == 'Charged Off':
                recommendations.extend(['Home Loan', 'Personal Loan'])
        if user_data['Personal Loan'].iloc[0] == 1 and loan_status == 'Fully Paid':
            if user_data['Credit Score'].iloc[0] < 700:
                recommendations.append('Vehicle Loan')
            elif user_data['Credit Score'].iloc[0] > 700:
                recommendations.extend(['Home Loan', 'Vehicle Loan'])
        if user_data['Credit Score'].iloc[0] > 700 and user_data['Home Ownership'].iloc[
            0] == 'Rent' and loan_status == 'Fully Paid':
            recommendations.append('Home Loan')
        else:
            recommendations.append('Shopping Loan')

        # Return unique recommendations
        recommendations = list(set(recommendations))
        response.append(
            f"\nBased on your profile, We Recommend the following loans: {', '.join(recommendations)}. ")
    else:
        response.append("The provided Customer Code is wrong. Please Check your Customer Code")

    return "\n".join(response)

def new_customer_code(sample_test_data):
    # Load the saved model and label encoders
    rf_classifier = joblib.load('rf_loan_model.pkl')
    le_term = joblib.load('le_term.pkl')
    le_home_ownership = joblib.load('le_home_ownership.pkl')
    le_loan_type = joblib.load('le_loan_type.pkl')
    le_loan_status = joblib.load('le_loan_status.pkl')

    def preprocess_data(data):
        """Convert categorical columns to numerical values using the label encoders."""
        data['Term'] = le_term.transform(data['Term'])
        data['Home Ownership'] = le_home_ownership.transform(data['Home Ownership'])
        data['Loan_type'] = le_loan_type.transform(data['Loan_type'])
        return data

    def predict_loan_status(data):
        """Make predictions using the Random Forest model and convert them to 'Eligible' or 'Not Eligible'."""
        preprocessed_data = preprocess_data(data)
        predictions = rf_classifier.predict(preprocessed_data)
        decoded_predictions = le_loan_status.inverse_transform(predictions)
        eligibility_results = ["Eligible" if prediction == "Fully Paid" else "Not Eligible" for prediction in decoded_predictions]
        return eligibility_results[0]

    # 1. Load the data
    merged_df = pd.read_excel("Merged_Loan&Recommender_data.xlsx")

    # 2. Data Preprocessing
    for column in ['Credit Score', 'household_income_x', 'Monthly Debt', 'Years of Credit History']:
        merged_df[column].fillna(merged_df[column].median(), inplace=True)
    for column in ['Term', 'Home Ownership', 'Loan_type', 'Loan Status']:
        merged_df[column].fillna(merged_df[column].mode()[0], inplace=True)

    label_encoders = {}
    for column in ['Term', 'Home Ownership', 'Loan_type', 'Loan Status']:
        le = LabelEncoder()
        merged_df[column] = le.fit_transform(merged_df[column])
        label_encoders[column] = le

    eligibility_result = predict_loan_status(sample_test_data)
    print(eligibility_result)

    if eligibility_result == 'Eligible':
        user_data = merged_df[['cust_code', 'Credit Score', 'household_income_x', 'Home Ownership', 'Loan Status']]
        sample_data = user_data.sample(5000, random_state=42)
        scaler = StandardScaler()
        sample_data_scaled = scaler.fit_transform(sample_data.drop('cust_code', axis=1))
        sample_similarity = cosine_similarity(sample_data_scaled)

        def recommend_loans(target_user, similarity_df, original_data, top_n=5):
            similar_users = similarity_df[target_user].sort_values(ascending=False).index[1:top_n + 1]
            loan_columns = ['Business Loan', 'Education Loan', 'Home Loan', 'Personal Loan', 'shopping loan',
                            'Vehicle Loan', 'wedding loan']
            similar_users_loans = original_data[original_data['cust_code'].isin(similar_users)][loan_columns]
            recommended_loans = similar_users_loans.sum().sort_values(ascending=False)
            target_user_loans = original_data[original_data['cust_code'] == target_user][loan_columns].iloc[0]
            recommended_loans = recommended_loans[target_user_loans == 0]
            return recommended_loans

        # Start the encoding process for 'home_ownership_value' here
        home_ownership_value = sample_test_data['Home Ownership'].iloc[0]
        try:
            home_ownership_encoded = label_encoders['Home Ownership'].transform([home_ownership_value])[0]
        except ValueError:
            print("Warning: Unseen label for Home Ownership. Assigning a default value.")
            home_ownership_encoded = -1

        #home_ownership_value = sample_test_data['Home Ownership'].iloc[0]
        new_user_data = pd.DataFrame({
        'Credit Score': [sample_test_data['Credit Score'].iloc[0]],
        'household_income_x': [sample_test_data['household_income'].iloc[0]],
        'Home Ownership': [home_ownership_encoded],
        'Loan Status': label_encoders['Loan Status'].transform(['Fully Paid'])  # Get Input from User
        })
        new_user_scaled = scaler.transform(new_user_data)
        new_user_similarity = cosine_similarity(new_user_scaled, sample_data_scaled)
        new_user_similarity_series = pd.Series(new_user_similarity[0], index=sample_data['cust_code'])
        sample_similarity_df = pd.DataFrame(sample_similarity, index=sample_data['cust_code'], columns=sample_data['cust_code'])
        recommended_loans_for_new_user = recommend_loans(new_user_similarity_series.idxmax(), sample_similarity_df, merged_df)

        # Filter out loans with a recommendation value of 0
        recommended_loans_filtered = recommended_loans_for_new_user[recommended_loans_for_new_user > 0]

        # Printing the recommended loans
        loan_names = []

        # If there are recommended loans, add them to the list
        if not recommended_loans_filtered.empty:
            for loan_name in recommended_loans_filtered.index:
                loan_names.append(loan_name)
        else:
            credit_score = new_user_data['Credit Score'].iloc[0]
            loan_status = label_encoders['Loan Status'].inverse_transform(new_user_data['Loan Status'])[0]
            home_ownership_value = sample_test_data['Home Ownership'].iloc[0]
            try:
                home_ownership_encoded = label_encoders['Home Ownership'].transform([home_ownership_value])[0]
            except ValueError:
                print("Warning: Unseen label for Home Ownership. Assigning a default value.")
                home_ownership_encoded = -1

            if loan_status == 'Fully Paid':
                if home_ownership_encoded == label_encoders['Home Ownership'].transform(['Rent'])[
                    0] and credit_score >= 700:
                    loan_names.append("Home Loan")
                elif home_ownership_encoded != label_encoders['Home Ownership'].transform(['Rent'])[
                    0] and credit_score >= 700:
                    loan_names.extend(["Personal Loan", "Vehicle Loan"])
                elif home_ownership_encoded != label_encoders['Home Ownership'].transform(['Rent'])[
                    0] and credit_score < 700:
                    loan_names.append("Personal Loan")

        return eligibility_result, loan_names

    elif eligibility_result == 'Not Eligible':
        credit_score = sample_test_data['Credit Score'].iloc[0]
        if credit_score >= 700:
            loan_names = ['Personal Loan']
        else:
            loan_names = ["Shopping loan"]

        return eligibility_result, loan_names

    #return loan_names

def main():
    while True: # Infinite loop to ensure user gives correct input or exits
        answer = input("Are you an Existing Customer (Enter 1) or New Customer (Enter 2)? ")

        if answer == "1":
            cust_code = int(input("Please Enter Your Customer Code: "))
            start = time.time()
            print(existing_customer_code(cust_code))
            end = time.time()
            print("\nTime taken to Predict Loan Decision and Recommend the Product for existing Customer:", end-start)
            break
        elif answer == "2":
            # Replace this with your new data as needed
            def get_numeric_input(prompt):
                """Function to get a numeric input from the user."""
                while True:
                    try:
                        value = int(input(prompt))
                        return value
                    except ValueError:
                        print("Please enter a valid numeric value.")

            def get_float_input(prompt):
                """Function to get a float input from the user."""
                while True:
                    try:
                        value = float(input(prompt))
                        return value
                    except ValueError:
                        print("Please enter a valid numeric value.")

            # Mapping for term, home ownership, and loan type
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

            # Get inputs from the user
            current_loan_amount = get_numeric_input("Enter the Loan Amount required: ")

            term = None
            while term not in term_mapping:
                term = get_numeric_input("Enter 1 for 'Short Term' and 2 for 'Long Term': ")
            term = term_mapping[term]

            credit_score = get_numeric_input("Enter your Credit Score: ")

            household_income = get_numeric_input("Enter your Income: ")

            years_in_current_job = get_numeric_input(
                "Enter the number of years of experience in your job (e.g. 1, 2, 3): ")

            home_ownership = None
            while home_ownership not in home_ownership_mapping:
                home_ownership = get_numeric_input("Enter 1 for 'Rent', 2 for 'Home Mortgage', and 3 for 'Own Home': ")
            home_ownership = home_ownership_mapping[home_ownership]

            loan_type = None
            while loan_type not in loan_type_mapping:
                loan_type = get_numeric_input(
                    "Enter 1 for 'Personal Loan', 2 for 'Home Loan', 3 for 'Business Loan', 4 for 'Education Loan', 5 for 'Vehicle Loan', 6 for 'shopping loan', and 7 for 'wedding loan': ")
            loan_type = loan_type_mapping[loan_type]

            monthly_debt = get_float_input("Enter your Monthly Debt: ")

            years_of_credit_history = get_float_input("Enter your Years of Credit History: ")

            sample_test_data = pd.DataFrame({
                'Current Loan Amount': [current_loan_amount],
                'Term': [term],
                'Credit Score': [credit_score],
                'household_income': [household_income],
                'Years in current job': [years_in_current_job],
                'Home Ownership': [home_ownership],
                'Loan_type': [loan_type],
                'Monthly Debt': [monthly_debt],
                'Years of Credit History': [years_of_credit_history]
            })

            #print(sample_test_data)
            start = time.time()
            print(new_customer_code(sample_test_data))
            end = time.time()
            print("Time taken to Predict Loan Decision and Recommend the Product: ",end-start)
            break
        else:
            print("You have entered a wrong input. Please Enter 1 for Existing Customer or 2 for New Customer.")

if __name__ == "__main__":
    main()
