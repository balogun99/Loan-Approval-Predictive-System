# load the libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# load the model
model = joblib.load('/Users/macbookpro/Desktop/STUDENT DROPOUT PREDICTION USING A RANDOM FOREST/model/rf_model.pkl')

st.title('Student Dropout Predictive System')
st.write('Enter Student Information to Predict Dropout Risk')

# User Inputs
age = st.number_input("Age at Enrollment", 16, 20)
admission_grade = st.number_input("Admission Grade", 0.0, 200.0)
units_approved_1 = st.number_input("1st Semester Approved Units", 0, 20)
units_grade_1 = st.number_input("1st Semester Grade", 0.0, 20.0)

tuition = st.selectbox(
    "Tuition Fees Up to Date",
    ["Yes", "No"]
)

scholarship = st.selectbox(
    "Scholarship Holder",
    ["Yes", "No"]
)

# convert selections
tuition = 1 if tuition == 'Yes' else 0
scholarship = 1 if scholarship == "Yes" else 0

# predict button
if st.button("Predict Dropout Risk"):
    input_data = np.array([[
        age,
        admission_grade,
        units_approved_1,
        units_grade_1,
        tuition,
        scholarship
    ]])

# make predictions based on the data
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)

if prediction[0] == 1:
    st.error("⚠️ Student is LIKELY to DROPOUT")
else:
    st.success("✅ Student is NOT LIKELY to DROPOUT")
    
st.write("Prediction Probababilty :", prob)

# not completed - remaining other features

# Marital status;Application mode;Application order;
# Course;"Daytime/evening attendance	
# ";Previous qualification;Previous qualification (grade)
# ;Nacionality;Mother's qualification;Father's qualification;
# Mother's occupation;Father's occupation;Admission grade;
# Displaced;Educational special needs;Debtor;Tuition fees up to date;
# Gender;Scholarship holder;Age at enrollment;International;
# Curricular units 1st sem (credited);Curricular units 1st sem (enrolled);
# Curricular units 1st sem (evaluations);Curricular units 1st sem (approved);
# Curricular units 1st sem (grade);Curricular units 1st sem (without evaluations);
# Curricular units 2nd sem (credited);Curricular units 2nd sem (enrolled);Curricular units 2nd sem (evaluations);
# Curricular units 2nd sem (approved);Curricular units 2nd sem (grade);Curricular units 2nd sem (without evaluations);
# Unemployment rate;Inflation rate;GDP;