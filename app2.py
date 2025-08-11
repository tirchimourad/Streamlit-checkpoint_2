import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle sauvegardé (exemple : SVM)
model = joblib.load("regression_logistic.pkl")

# Charger aussi le scaler
scaler = joblib.load("scaler.pkl")

st.title("Prédiction de possession de compte bancaire")
st.write("Remplissez les informations ci-dessous pour savoir si une personne possède un compte bancaire.")

# Champs de saisie 
country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Année", [2016, 2017, 2018])
location_type = st.selectbox("Type de lieu", ["Rural", "Urban"])
cellphone_access = st.selectbox("Accès téléphone", ["Yes", "No"])
household_size = st.number_input("Taille du foyer", min_value=1, max_value=50, value=5)
age_of_respondent = st.number_input("Âge du répondant", min_value=15, max_value=100, value=30)
gender_of_respondent = st.selectbox("Genre", ["Male", "Female"])
relationship_with_head = st.selectbox("Relation avec le chef du foyer", ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])
marital_status = st.selectbox("Statut marital", ["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Separated", "Dont know"])
education_level = st.selectbox("Niveau d'éducation", ["No formal education", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])
job_type = st.selectbox("Type d'emploi", ["Self employed", "Government Dependent", "Formally employed Private", "Formally employed Government", "Informally employed", "Farming and Fishing", "Remittance Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"])

# Bouton prédiction
if st.button("Prédire"):
    # Créer un DataFrame pour la ligne saisie
    input_data = pd.DataFrame([{
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head,
        "marital_status": marital_status,
        "education_level": education_level,
        "job_type": job_type
    }])

    # Encodage identique à l'entraînement
    input_encoded = pd.get_dummies(input_data)
    
    # Recharger l'ordre des colonnes utilisé à l'entraînement
    columns_train = joblib.load("columns.pkl")
    input_encoded = input_encoded.reindex(columns=columns_train, fill_value=0)

    # Mise à l'échelle
    input_scaled = scaler.transform(input_encoded)

    # Prédiction
    prediction = model.predict(input_scaled)[0]

    if prediction == "Yes":
        st.success("✅ La personne possède probablement un compte bancaire.")
    else:
        st.error("❌ La personne ne possède probablement pas de compte bancaire.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle sauvegardé (exemple : SVM)
model = joblib.load("regression_logistic.pkl")

# Charger aussi le scaler
scaler = joblib.load("scaler.pkl")

st.title("Prédiction de possession de compte bancaire")
st.write("Remplissez les informations ci-dessous pour savoir si une personne possède un compte bancaire.")

# Champs de saisie 
country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Année", [2016, 2017, 2018])
location_type = st.selectbox("Type de lieu", ["Rural", "Urban"])
cellphone_access = st.selectbox("Accès téléphone", ["Yes", "No"])
household_size = st.number_input("Taille du foyer", min_value=1, max_value=50, value=5)
age_of_respondent = st.number_input("Âge du répondant", min_value=15, max_value=100, value=30)
gender_of_respondent = st.selectbox("Genre", ["Male", "Female"])
relationship_with_head = st.selectbox("Relation avec le chef du foyer", ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])
marital_status = st.selectbox("Statut marital", ["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Separated", "Dont know"])
education_level = st.selectbox("Niveau d'éducation", ["No formal education", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])
job_type = st.selectbox("Type d'emploi", ["Self employed", "Government Dependent", "Formally employed Private", "Formally employed Government", "Informally employed", "Farming and Fishing", "Remittance Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"])

# Bouton prédiction
if st.button("Prédire"):
    # Créer un DataFrame pour la ligne saisie
    input_data = pd.DataFrame([{
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head,
        "marital_status": marital_status,
        "education_level": education_level,
        "job_type": job_type
    }])

    # Encodage identique à l'entraînement
    input_encoded = pd.get_dummies(input_data)
    
    # Recharger l'ordre des colonnes utilisé à l'entraînement
    columns_train = joblib.load("columns.pkl")
    input_encoded = input_encoded.reindex(columns=columns_train, fill_value=0)

    # Mise à l'échelle
    input_scaled = scaler.transform(input_encoded)

    # Prédiction
    prediction = model.predict(input_scaled)[0]

    if prediction == "Yes":
        st.success("✅ La personne possède probablement un compte bancaire.")
    else:
        st.error("❌ La personne ne possède probablement pas de compte bancaire.")


