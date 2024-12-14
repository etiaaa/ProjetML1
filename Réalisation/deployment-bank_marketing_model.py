import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle entraîné
try:
    model = joblib.load("./bank_marketing_model.pkl")
    st.sidebar.success("Modèle chargé avec succès!")
except FileNotFoundError:
    st.sidebar.error("Le fichier du modèle est introuvable. Veuillez vérifier le chemin.")
    st.stop()

# Titre de l'application
st.title("Prédiction d'acceptation d'offre de dépôt à terme")

# Description
st.write("""
Cette application prédit si un client acceptera une offre de dépôt à terme,
basée sur ses informations personnelles et historiques.
""")

# Fonction pour saisir les données utilisateur
def get_user_input():
    st.sidebar.header("Saisir les données utilisateur")

    age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30, step=1)
    job = st.sidebar.selectbox("Profession", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                      'management', 'retired', 'self-employed', 'services',
                                      'student', 'technician', 'unemployed', 'unknown'])
    marital = st.sidebar.selectbox("Statut marital", ['married', 'single', 'divorced', 'unknown'])
    education = st.sidebar.selectbox("Niveau d'éducation", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.sidebar.selectbox("Défaut de crédit", ['yes', 'no', 'unknown'])
    housing = st.sidebar.selectbox("Prêt immobilier", ['yes', 'no', 'unknown'])
    loan = st.sidebar.selectbox("Prêt personnel", ['yes', 'no', 'unknown'])
    contact = st.sidebar.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])
    month = st.sidebar.selectbox("Mois de contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                             'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.sidebar.selectbox("Jour de la semaine", ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.sidebar.number_input("Durée du contact (en secondes)", min_value=0, max_value=5000, value=200, step=1)
    campaign = st.sidebar.number_input("Nombre de contacts lors de cette campagne", min_value=1, max_value=50, value=1, step=1)
    pdays = st.sidebar.number_input("Jours depuis la dernière campagne (999 si aucun contact)", min_value=0, max_value=999, value=999, step=1)
    previous = st.sidebar.number_input("Nombre de contacts avant cette campagne", min_value=0, max_value=50, value=0, step=1)
    poutcome = st.sidebar.selectbox("Résultat de la campagne précédente", ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.sidebar.number_input("Taux de variation de l'emploi (%)", min_value=-10.0, max_value=10.0, value=1.1, step=0.1)
    cons_price_idx = st.sidebar.number_input("Indice des prix à la consommation", min_value=90.0, max_value=100.0, value=93.994, step=0.001)
    cons_conf_idx = st.sidebar.number_input("Indice de confiance des consommateurs", min_value=-50.0, max_value=0.0, value=-36.4, step=0.1)
    euribor3m = st.sidebar.number_input("Taux Euribor 3 mois (%)", min_value=0.0, max_value=10.0, value=4.857, step=0.001)
    nr_employed = st.sidebar.number_input("Nombre d'employés", min_value=0.0, max_value=10000.0, value=5191.0, step=0.1)

    # Rassembler les entrées dans un DataFrame
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }
    return pd.DataFrame([data])

# Obtenir les données utilisateur
user_input = get_user_input()

# Afficher les données utilisateur
st.subheader("Données saisies par l'utilisateur")
st.write(user_input)

# Faire une prédiction
if st.button("Prédire"):
    try:
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)[0]
        if prediction[0] == 1:
            st.success(f"Le client est susceptible d'accepter l'offre. (Probabilité: {prediction_proba[1] * 100:.2f}%)")
        else:
            st.warning(f"Le client est peu susceptible d'accepter l'offre. (Probabilité: {prediction_proba[0] * 100:.2f}%)")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
