import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the survey data
data = pd.read_csv('Vaccinaion Update.csv')

# Convert 'Vaccination_Intention' to numerical values
data['Vaccine Status'] = data['Vaccine Status'].apply(lambda x: 1 if x == 'Yes' else 0)
data.drop('Which of these activities below affected your behaviour  during the pandemic period ', axis=1,inplace=True)
# Convert other categorical variables to numerical values using get_dummies or LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Select features and target variable
X = data.drop('Vaccine Status', axis=1)
y = data['Vaccine Status']

# Load the trained model from the file
with open('vaccination_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the scaler from the file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Standardize the features
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# Predict the probability of vaccination for the entire dataset using the loaded model
y_prob = loaded_model.predict_proba(X_scaled)[:, 1]

# Assume you have the total population size
total_population = 3551000

# Calculate the expected number of people to get vaccinated
predicted_vaccinated = np.mean(y_prob) * total_population

# Display the result with Streamlit
st.title("Vaccination Prediction")
st.write(f'Predicted number of people to get vaccinated: {predicted_vaccinated:.0f}')
