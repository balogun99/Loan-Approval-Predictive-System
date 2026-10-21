# import libraries
# from narwhals import col
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load the model
# model = joblib.load('/Users/macbookpro/Desktop/ML & DL & GENAI Projects/Loan Approval Prediction/loan_approval_model.sav')
model = joblib.load('lr_loan_approval_model.sav')

# app title
st.title("Loan Approval Predictive System")
st.write("Enter your details for loan approval")

# user inputs
col1, col2, col3 = st.columns(3)

with col1:
    Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
    # Gender = st.number_input("Age (0 - Female & 1 - Male)", 0,1)

with col2:
    Married = st.selectbox(
    "Married",
    ["Yes", "No"]
)
    # Married = st.number_input("Married (0 - No & 1 - Yes)", 0,1)

with col3:
    Education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)
    
with col1:
    Property_Area = st.selectbox(
        "Property_Area",
        ["Urban", "Rural", "Semiurban"]
    )

with col2:
    Dependents = st.number_input("Dependents")
    # Education = st.number_input("Education")

with col3:
    Self_Employed = st.number_input("Self-Employed")

with col1:
    ApplicantIncome = st.number_input("Applicant Income")

with col2:
    CoapplicantIncome = st.number_input("Co-Applicant Income")

with col3:
    LoanAmount = st.number_input("Loan Amount")

with col1:
    Loan_Amount_Term = st.number_input("Loan Amount Term")

with col2:
    Credit_History = st.number_input("Credit History")

with col3:
    Property_Area = st.number_input("Property Area")

# convert selections
Gender = 1 if Gender == 'Yes' else 0
Married = 0 if Married == 'No' else 1
Education = 0 if Education == 'Graduate' else 1
Self_Employed = 0 if Self_Employed == 'No' else 1
Property_Area = 0 if Property_Area == 'No' else 1

input_data = np.array([[
        Gender,	
        Married, 
        Dependents,	
        Education,	
        Self_Employed,	
        ApplicantIncome, 
        CoapplicantIncome,	
        LoanAmount,	
        Loan_Amount_Term,	
        Credit_History,	
        Property_Area
    ]])

if st.button("Predict Loan Approval"):
    input_data = np.array([[
        Gender,	
        Married, 
        Dependents,	
        Education,	
        Self_Employed,	
        ApplicantIncome, 
        CoapplicantIncome,	
        LoanAmount,	
        Loan_Amount_Term,	
        Credit_History,	
        Property_Area
    ]])

# make predictions
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)

if prediction[0] == 1:
    # prediction = "Loan NOT Approved"
    st.error("Loan NOT Approved")
else:
    # prediction = "Loan NOT Approved"
    st.success("Loan APPROVED")

st.write("Prediction Probability :", prob)

# Gender	
# Married	Dependents	Education	Self_Employed	
# ApplicantIncome	CoapplicantIncome	LoanAmount	
# Loan_Amount_Term	Credit_History	Property_Area	