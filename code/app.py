import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('../model/random_forest_model.pkl')

# Streamlit App UI
st.title("Customer Churn Prediction App")
st.write("Predict the likelihood of customer churn based on account and usage information. Fill in the fields below and click 'Predict Churn' to get the result.")

# Define the input fields with improved labels and descriptions
account_length = st.number_input("Account Duration (in days)", min_value=1, max_value=500, value=100,
                                 help="Total duration (in days) that the customer has been with the company.")
area_code = st.selectbox("Customer's Area Code", [408, 415, 510], 
                         help="Area code associated with the customer’s phone number.")
international_plan = st.selectbox("International Plan Subscription", ['Yes', 'No'], 
                                  help="Indicates if the customer is subscribed to an international calling plan.")
voice_mail_plan = st.selectbox("Voice Mail Plan Subscription", ['Yes', 'No'], 
                               help="Indicates if the customer is subscribed to a voice mail plan.")
number_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0, value=10,
                                        help="The total number of voice mail messages received by the customer.")
customer_service_calls = st.number_input("Customer Service Calls Made", min_value=0, value=1,
                                         help="The number of times the customer has called customer service.")
total_charge = st.number_input("Total Monthly Charges (USD)", min_value=0.0, value=50.0,
                               help="Total charges for the customer’s account, including day, evening, night, and international charges.")
total_usage = st.number_input("Total Call Usage (in minutes)", min_value=0.0, value=300.0,
                              help="The total minutes of calls made by the customer (day, evening, night, and international combined).")
total_calls = st.number_input("Total Number of Calls Made", min_value=0, value=50,
                              help="The total number of calls made by the customer (day, evening, night, and international combined).")

# Map categorical inputs to numerical values (if necessary)
international_plan = 1 if international_plan == 'Yes' else 0
voice_mail_plan = 1 if voice_mail_plan == 'Yes' else 0

# Prepare the input data as a DataFrame (for the model)
input_data = pd.DataFrame({
    'account length': [account_length],
    'area code': [area_code],
    'international plan': [international_plan],
    'voice mail plan': [voice_mail_plan],
    'number vmail messages': [number_vmail_messages],
    'customer service calls': [customer_service_calls],
    'total charge': [total_charge],
    'total_usage': [total_usage],
    'total_calls': [total_calls]
})

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction == 1:
        st.write("### Prediction: This customer is likely to churn.")
        st.write("The model predicts that the customer is at risk of churning.")
    else:
        st.write("### Prediction: This customer is not likely to churn.")
        st.write("The model predicts that the customer is not at risk of churning.")