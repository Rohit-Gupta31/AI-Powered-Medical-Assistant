import streamlit as st
import numpy as np
import pandas as pd
import pickle
import ast

model = pickle.load(open('model.pkl', 'rb'))
model_data = pickle.load(open('model_data.pkl', 'rb'))
symptoms_dict = model_data['symptom_dict']
diseases_list = model_data['diseases_list']


precautions = pd.read_csv('precautions_df.csv')
diets = pd.read_csv('diets.csv')
workouts = pd.read_csv('workout_df.csv')
descriptions = pd.read_csv('description.csv')
medications = pd.read_csv('medications.csv')


st.title("👩🏻‍⚕️ Your AI-Powered Doctor")
st.markdown("Enter your symptoms to get a diagnosis and care suggestions.")


available_symptoms = list(symptoms_dict.keys())
patient_symptoms = st.multiselect(
    "Select your symptoms:",
    options=sorted(available_symptoms)
)


def get_prediction(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return diseases_list[model.predict([input_vector])[0]]

patient_disease = get_prediction(patient_symptoms)

def recommendations(patient_disease):
    precaution_info = precautions[precautions['Disease'] == patient_disease]
    if not precaution_info.empty:
        precaution_list = precaution_info.iloc[0, 2:].dropna().tolist()
        precaution_list = ' , '.join(precaution_list)
    else:
        precaution_list = "No precaution info available"

    diet_info = diets[diets['Disease'] == patient_disease]
    if not diet_info.empty:
        diet_list = diet_info.iloc[0]['Diet']
        diet_list = ast.literal_eval(diet_list)
        diet_list = ' , '.join(diet_list)
    else:
        diet_list = "No diet info available"

    workout_info = workouts[workouts['disease'] == patient_disease]
    if not workout_info.empty:
        workout_list = workout_info['workout'].dropna().tolist()
        workout_list = ' , '.join(workout_list)  # Combine into one string if you prefer
    else:
        workout_list = "No workout recommendations available"

    description_info = descriptions[descriptions['Disease'] == patient_disease]
    if not description_info.empty:
        description_list = description_info.iloc[0]['Description']
    else:
        description_list = "No description info available"

    medication_info = medications[medications['Disease'] == patient_disease]
    if not medication_info.empty:
        medication_list = medication_info.iloc[0]['Medication']
        medication_list = ast.literal_eval(medication_list)
        medication_list = ' , '.join(medication_list)
    else:
        medication_list = "No medication info available"

    return precaution_list, diet_list, workout_list, description_list, medication_list


precaution, diet, workout, description, medication = recommendations(patient_disease)

st.write("---")
if st.button("⚕️ Disease Name"):
    st.info(patient_disease)

if st.button("🧾 Description"):
    st.write(description)

if st.button("🛡️ Precautions"):
    st.write(precaution)

if st.button("🥗 Diet"):
    st.write(diet)

if st.button("🏋️ Workout"):
    st.write(workout)

if st.button("💊 Medications"):
    st.write(medication)
