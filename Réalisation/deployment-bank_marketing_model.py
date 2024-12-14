import streamlit as st  # Framework pour créer des interfaces utilisateur interactives
import pandas as pd  # Bibliothèque pour manipuler les données tabulaires
import numpy as np  # Bibliothèque pour effectuer des calculs numériques
import joblib  # Pour sauvegarder et charger des modèles et objets Python

# Charger le modèle entraîné
try:
    model = joblib.load("./bank_marketing_model.pkl")  # Charger le modèle préalablement sauvegardé
    st.sidebar.success("Modèle chargé avec succès!")  # Afficher un message de succès si le modèle est chargé
except FileNotFoundError:
    st.sidebar.error("Le fichier du modèle est introuvable. Veuillez vérifier le chemin.")  # Gérer l'absence du fichier
    st.stop()  # Arrêter l'exécution de l'application si le modèle est introuvable

# Titre de l'application
st.title("Prédiction d'acceptation d'offre de dépôt à terme")  # Titre principal de l'application

# Description
st.write("""
Cette application prédit si un client acceptera une offre de dépôt à terme,
basée sur ses informations personnelles et historiques.
""")  # Description de l'objectif de l'application

# Fonction pour saisir les données utilisateur
def get_user_input():
    st.sidebar.header("Saisir les données utilisateur")  # Section d'entrée utilisateur dans la barre latérale

    # Collecter les données utilisateur à travers des champs interactifs
    age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30, step=1)  # Entrée pour l'âge
    job = st.sidebar.selectbox("Profession", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                      'management', 'retired', 'self-employed', 'services',
                                      'student', 'technician', 'unemployed', 'unknown'])  # Sélection de la profession
    marital = st.sidebar.selectbox("Statut marital", ['married', 'single', 'divorced', 'unknown'])  # Statut marital
    education = st.sidebar.selectbox("Niveau d'éducation", ['primary', 'secondary', 'tertiary', 'unknown'])  # Niveau éducatif
    default = st.sidebar.selectbox("Défaut de crédit", ['yes', 'no', 'unknown'])  # Historique de défaut de crédit
    housing = st.sidebar.selectbox("Prêt immobilier", ['yes', 'no', 'unknown'])  # Statut du prêt immobilier
    loan = st.sidebar.selectbox("Prêt personnel", ['yes', 'no', 'unknown'])  # Statut du prêt personnel
    contact = st.sidebar.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])  # Méthode de contact
    month = st.sidebar.selectbox("Mois de contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                             'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])  # Mois de l'interaction
    day_of_week = st.sidebar.selectbox("Jour de la semaine", ['mon', 'tue', 'wed', 'thu', 'fri'])  # Jour de la semaine
    duration = st.sidebar.number_input("Durée du contact (en secondes)", min_value=0, max_value=5000, value=200, step=1)  # Durée de l'appel
    campaign = st.sidebar.number_input("Nombre de contacts lors de cette campagne", min_value=1, max_value=50, value=1, step=1)  # Nombre de contacts
    pdays = st.sidebar.number_input("Jours depuis la dernière campagne (999 si aucun contact)", min_value=0, max_value=999, value=999, step=1)  # Délai depuis le dernier contact
    previous = st.sidebar.number_input("Nombre de contacts avant cette campagne", min_value=0, max_value=50, value=0, step=1)  # Nombre de contacts antérieurs
    poutcome = st.sidebar.selectbox("Résultat de la campagne précédente", ['failure', 'nonexistent', 'success'])  # Résultat précédent
    emp_var_rate = st.sidebar.number_input("Taux de variation de l'emploi (%)", min_value=-10.0, max_value=10.0, value=1.1, step=0.1)  # Taux de variation de l'emploi
    cons_price_idx = st.sidebar.number_input("Indice des prix à la consommation", min_value=90.0, max_value=100.0, value=93.994, step=0.001)  # Indice des prix à la consommation
    cons_conf_idx = st.sidebar.number_input("Indice de confiance des consommateurs", min_value=-50.0, max_value=0.0, value=-36.4, step=0.1)  # Confiance des consommateurs
    euribor3m = st.sidebar.number_input("Taux Euribor 3 mois (%)", min_value=0.0, max_value=10.0, value=4.857, step=0.001)  # Taux Euribor 3 mois
    nr_employed = st.sidebar.number_input("Nombre d'employés", min_value=0.0, max_value=10000.0, value=5191.0, step=0.1)  # Nombre d'employés

    # Rassembler les données dans un dictionnaire et les convertir en DataFrame
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
    return pd.DataFrame([data])  # Retourner les données utilisateur sous forme de DataFrame

# Obtenir les données utilisateur
user_input = get_user_input()  # Appeler la fonction pour collecter les données utilisateur

# Afficher les données utilisateur
st.subheader("Données saisies par l'utilisateur")  # Sous-titre pour les données utilisateur
st.write(user_input)  # Afficher les données sous forme de tableau

# Faire une prédiction
if st.button("Prédire"):  # Bouton pour lancer la prédiction
    try:
        prediction = model.predict(user_input)  # Prédire la classe (0 ou 1)
        prediction_proba = model.predict_proba(user_input)[0]  # Obtenir les probabilités des classes
        if prediction[0] == 1:
            st.success(f"Le client est susceptible d'accepter l'offre à {prediction_proba[1] * 100:.2f}%")  # Résultat positif
        else:
            st.warning(f"Le client est susceptible de refuser l'offre à {prediction_proba[0] * 100:.2f}%")  # Résultat négatif
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")  # Afficher un message d'erreur si nécessaire
