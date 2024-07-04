import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Custom label encoding function
def custom_label_encoder(column):
    categories = column.unique()
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    return column.map(mapping)

# Load the survey data
data = pd.read_csv('Vaccinaion Update.csv')

# Convert 'Vaccination_Intention' to numerical values
data['Vaccine Status'] = data['Vaccine Status'].apply(lambda x: 1 if x == 'Yes' else 0)
data.drop('Which of these activities below affected your behaviour  during the pandemic period ', axis=1,inplace=True)

# Convert other categorical variables to numerical values using custom label encoder
for column in data.select_dtypes(include=['object']).columns:
    data[column] = custom_label_encoder(data[column])

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

# Add styling and interactive elements to the Streamlit app
st.title("Vaccination Prediction")
st.markdown("""
<style>
.big-font {
    font-size: 36px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Predicted number of people to get vaccinated:</p>', unsafe_allow_html=True)
st.markdown(f'<p class="big-font">{predicted_vaccinated:.0f}</p>', unsafe_allow_html=True)


# Add a slider to adjust the total population size
total_population_slider = st.slider('Total Population Size', min_value=100000, max_value=5000000, value=3551000, step=10000)

# Calculate the expected number of people to get vaccinated based on the slider value
predicted_vaccinated_adjusted = np.mean(y_prob) * total_population_slider

# Display the result with Streamlit
st.title("Vaccination Prediction")
st.markdown('<p class="big-font">Predicted number of people to get vaccinated:</p>', unsafe_allow_html=True)
st.markdown(f'<p class="big-font">{predicted_vaccinated_adjusted:.0f}</p>', unsafe_allow_html=True)
