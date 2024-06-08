from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)

# Load pre-trained model and tokenizer
model = None
tokenizer = None
disease_to_label = None
merged_df = None
disease_to_label = {'Diabetes ': 0,
 'Hyperthyroidism': 1,
 'Malaria': 2,
 'Hypertension ': 3,
 'hepatitis A': 4,
 'AIDS': 5,
 'Gastroenteritis': 6,
 'GERD': 7,
 'Hepatitis E': 8,
 'Paralysis (brain hemorrhage)': 9,
 'Typhoid': 10,
 'Allergy': 11,
 'Heart attack': 12,
 'Varicose veins': 13,
 'Tuberculosis': 14,
 'Migraine': 15,
 'Jaundice': 16,
 'Hepatitis C': 17,
 'Osteoarthristis': 18,
 'Acne': 19,
 'Urinary tract infection': 20,
 'Bronchial Asthma': 21,
 'Hepatitis D': 22,
 'Common Cold': 23,
 'Pneumonia': 24,
 'Chronic cholestasis': 25,
 'Hypoglycemia': 26,
 '(vertigo) Paroymsal  Positional Vertigo': 27,
 'Alcoholic hepatitis': 28,
 'Arthritis': 29,
 'Impetigo': 30,
 'Hypothyroidism': 31,
 'Hepatitis B': 32,
 'Fungal infection': 33,
 'Dimorphic hemmorhoids(piles)': 34,
 'Chicken pox': 35,
 'Cervical spondylosis': 36,
 'Psoriasis': 37,
 'Drug Reaction': 38,
 'Dengue': 39,
 'Peptic ulcer diseae': 40}

def predict_disease(symptoms_list):
    cleaned_symptoms = [symptom.replace('_', ' ') if isinstance(symptom, str) else symptom for symptom in symptoms_list]
    cleaned_symptoms = [' '.join(map(str, symptom_list)) for symptom_list in cleaned_symptoms]
    sequences = tokenizer.texts_to_sequences(cleaned_symptoms)
    padded_sequences = pad_sequences(sequences, maxlen=30, padding='post')
    predictions = model.predict(padded_sequences)
    predicted_label_idx = np.argmax(predictions, axis=1)[0]

    predicted_disease = list(disease_to_label.keys())[list(disease_to_label.values()).index(predicted_label_idx)]
    return predicted_disease


def load_model_and_data():
    global model, tokenizer, disease_to_label, merged_df, dvd
    
    try:
        with open("custom_transformer.json", "r") as json_file:
            model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
        model.load_weights("custom_transformer_weights.h5")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    merged_df = pd.read_csv("C:\\Users\\Adhiraj\\css\\merged_doctor_specialization.csv", encoding='ISO-8859-1')
    dvd = pd.read_csv("C:\\Users\\Adhiraj\\Downloads\\Doctor_Versus_Disease (1).csv", encoding='ISO-8859-1', names=['Disease', 'Specialist'])


load_model_and_data()


@app.route('/predict_disease_and_doctor', methods=['POST'])
def predict_disease_and_doctor():
    symptoms_list = request.json.get('symptoms', [])
    predicted_disease = predict_disease(symptoms_list)
    # Fetch doctor's specialization
    doctor_specialization = dvd[dvd["Disease"] == predicted_disease]["Specialist"].values[0]

    # Fetch doctor's name from merged_df based on specialization
    filtered_doctors = merged_df[merged_df["Specialist"] == doctor_specialization]["Name"].str.replace('\xa0', '').tolist()
    if any(pd.isna(doc) for doc in filtered_doctors):
        filtered_doctors = ['General Phyisican']
    
    return jsonify({
        "predicted_disease": predicted_disease,
        "doctor_specialization": doctor_specialization,
        "filtered_doctors": filtered_doctors
    })

if __name__ == '__main__':
    app.run(debug=True)
