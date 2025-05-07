from flask import Flask, request, jsonify
import pandas as pd
import random
import torch
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

new_lst = [{'symptoms': 'itching, skin rash, nodal skin eruptions, dischromic  patches',
  'disease': 'Fungal infection'},
 {'symptoms': ' continuous sneezing, shivering, chills, watering from eyes',
  'disease': 'Allergy'},
 {'symptoms': ' stomach pain, acidity, ulcers on tongue, vomiting, cough, chest pain',
  'disease': 'GERD'},
 {'symptoms': 'itching, vomiting, yellowish skin, nausea, loss of appetite, abdominal pain, yellowing of eyes',
  'disease': 'Chronic cholestasis'},
 {'symptoms': 'itching, skin rash, stomach pain, burning micturition, spotting  urination',
  'disease': 'Drug Reaction'},
 {'symptoms': ' vomiting, indigestion, loss of appetite, abdominal pain, passage of gases, internal itching',
  'disease': 'Peptic ulcer diseae'},
 {'symptoms': ' muscle wasting, patches in throat, high fever, extra marital contacts',
  'disease': 'AIDS'},
 {'symptoms': ' fatigue, weight loss, restlessness, lethargy, irregular sugar level, blurred and distorted vision, obesity, excessive hunger, increased appetite, polyuria',
  'disease': 'Diabetes '},
 {'symptoms': ' vomiting, sunken eyes, dehydration, diarrhoea',
  'disease': 'Gastroenteritis'},
 {'symptoms': ' fatigue, cough, high fever, breathlessness, family history, mucoid sputum',
  'disease': 'Bronchial Asthma'},
 {'symptoms': ' headache, chest pain, dizziness, loss of balance, lack of concentration',
  'disease': 'Hypertension '},
 {'symptoms': ' acidity, indigestion, headache, blurred and distorted vision, excessive hunger, stiff neck, depression, irritability, visual disturbances',
  'disease': 'Migraine'},
 {'symptoms': ' back pain, weakness in limbs, neck pain, dizziness, loss of balance',
  'disease': 'Cervical spondylosis'},
 {'symptoms': ' vomiting, headache, weakness of one body side, altered sensorium',
  'disease': 'Paralysis (brain hemorrhage)'},
 {'symptoms': 'itching, vomiting, fatigue, weight loss, high fever, yellowish skin, dark urine, abdominal pain',
  'disease': 'Jaundice'},
 {'symptoms': ' chills, vomiting, high fever, sweating, headache, nausea, diarrhoea, muscle pain',
  'disease': 'Malaria'},
 {'symptoms': 'itching, skin rash, fatigue, lethargy, high fever, headache, loss of appetite, mild fever, swelled lymph nodes, malaise, red spots over body',
  'disease': 'Chicken pox'},
 {'symptoms': ' skin rash, chills, joint pain, vomiting, fatigue, high fever, headache, nausea, loss of appetite, pain behind the eyes, back pain, malaise, muscle pain, red spots over body',
  'disease': 'Dengue'},
 {'symptoms': ' chills, vomiting, fatigue, high fever, headache, nausea, constipation, abdominal pain, diarrhoea, toxic look (typhos), belly pain',
  'disease': 'Typhoid'},
 {'symptoms': ' joint pain, vomiting, yellowish skin, dark urine, nausea, loss of appetite, abdominal pain, diarrhoea, mild fever, yellowing of eyes, muscle pain',
  'disease': 'hepatitis A'},
 {'symptoms': 'itching, fatigue, lethargy, yellowish skin, dark urine, loss of appetite, abdominal pain, yellow urine, yellowing of eyes, malaise, receiving blood transfusion, receiving unsterile injections',
  'disease': 'Hepatitis B'},
 {'symptoms': ' fatigue, yellowish skin, nausea, loss of appetite, yellowing of eyes, family history',
  'disease': 'Hepatitis C'},
 {'symptoms': ' joint pain, vomiting, fatigue, yellowish skin, dark urine, nausea, loss of appetite, abdominal pain, yellowing of eyes',
  'disease': 'Hepatitis D'},
 {'symptoms': ' joint pain, vomiting, fatigue, high fever, yellowish skin, dark urine, nausea, loss of appetite, abdominal pain, yellowing of eyes, acute liver failure, coma, stomach bleeding',
  'disease': 'Hepatitis E'},
 {'symptoms': ' vomiting, yellowish skin, abdominal pain, swelling of stomach, distention of abdomen, history of alcohol consumption, fluid overload',
  'disease': 'Alcoholic hepatitis'},
 {'symptoms': ' chills, vomiting, fatigue, weight loss, cough, high fever, breathlessness, sweating, loss of appetite, mild fever, yellowing of eyes, swelled lymph nodes, malaise, phlegm, chest pain, blood in sputum',
  'disease': 'Tuberculosis'},
 {'symptoms': ' continuous sneezing, chills, fatigue, cough, high fever, headache, swelled lymph nodes, malaise, phlegm, throat irritation, redness of eyes, sinus pressure, runny nose, congestion, chest pain, loss of smell, muscle pain',
  'disease': 'Common Cold'},
 {'symptoms': ' chills, fatigue, cough, high fever, breathlessness, sweating, malaise, phlegm, chest pain, fast heart rate, rusty sputum',
  'disease': 'Pneumonia'},
 {'symptoms': ' constipation, pain during bowel movements, pain in anal region, bloody stool, irritation in anus',
  'disease': 'Dimorphic hemmorhoids(piles)'},
 {'symptoms': ' vomiting, breathlessness, sweating, chest pain',
  'disease': 'Heart attack'},
 {'symptoms': ' fatigue, cramps, bruising, obesity, swollen legs, swollen blood vessels, prominent veins on calf',
  'disease': 'Varicose veins'},
 {'symptoms': ' fatigue, weight gain, cold hands and feets, mood swings, lethargy, dizziness, puffy face and eyes, enlarged thyroid, brittle nails, swollen extremeties, depression, irritability, abnormal menstruation',
  'disease': 'Hypothyroidism'},
 {'symptoms': ' fatigue, mood swings, weight loss, restlessness, sweating, diarrhoea, fast heart rate, excessive hunger, muscle weakness, irritability, abnormal menstruation',
  'disease': 'Hyperthyroidism'},
 {'symptoms': ' vomiting, fatigue, anxiety, sweating, headache, nausea, blurred and distorted vision, excessive hunger, drying and tingling lips, slurred speech, irritability, palpitations',
  'disease': 'Hypoglycemia'},
 {'symptoms': ' joint pain, neck pain, knee pain, hip joint pain, swelling joints, painful walking',
  'disease': 'Osteoarthristis'},
 {'symptoms': ' muscle weakness, stiff neck, swelling joints, movement stiffness, painful walking',
  'disease': 'Arthritis'},
 {'symptoms': ' vomiting, headache, nausea, spinning movements, loss of balance, unsteadiness',
  'disease': '(vertigo) Paroymsal  Positional Vertigo'},
 {'symptoms': ' skin rash, pus filled pimples, blackheads, scurring',
  'disease': 'Acne'},
 {'symptoms': ' burning micturition, bladder discomfort, foul smell of urine, continuous feel of urine',
  'disease': 'Urinary tract infection'},
 {'symptoms': ' skin rash, joint pain, skin peeling, silver like dusting, small dents in nails, inflammatory nails',
  'disease': 'Psoriasis'},
 {'symptoms': ' skin rash, high fever, blister, red sore around nose, yellow crust ooze',
  'disease': 'Impetigo'}]

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoTokenizer

# Custom Transformer Model for Disease Classification
class CustomTransformerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=41):
        super(CustomTransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Feed inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Get the pooled output for classification
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits = self.classifier(pooled_output)  # Pass through the classifier
        return logits


df = pd.read_csv(r"C:\Users\Adhiraj\css\Transformer\merged_doctor_specialization.csv")
df = df.fillna('General Phyisican')
lst = []
CORS(app)
# Load your model
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Adhiraj\Downloads\tokenizer_path")
model = CustomTransformerModel()
model.load_state_dict(torch.load(r"C:\Users\Adhiraj\Downloads\custom_transformer_model.pth"))  # Load the saved state
model.eval()
st = set()

def predict_disease(passage):
  def preprocess_input(passage, tokenizer):
      # Tokenize the input symptoms
      tokenized_input = tokenizer(
          passage,
          padding=True,
          truncation=True,
          max_length=128,
          return_tensors="pt"  # Return PyTorch tensors
      )
      return tokenized_input['input_ids'], tokenized_input['attention_mask']

  input_ids, attention_mask = preprocess_input(passage, tokenizer)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)
  with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)

      probabilities = torch.sigmoid(outputs)  
      threshold = 0.5
      predicted_indices = (probabilities > threshold).nonzero(as_tuple=True)[1]  
      predicted_diseases = [new_lst[idx.item()]['disease'] for idx in predicted_indices]

  return predicted_diseases

def predict_disease_and_doctor(symptoms_list, merged_df):
    # Predict the disease based on the input symptoms
    predicted_disease = predict_disease(symptoms_list)
    doctor_specialization = set()
    lst = []
    for disease in predicted_disease:
        specialist = (merged_df.loc[merged_df["Disease_x"] == disease, "Specialist"].values)
        
        if specialist.size > 0:
            doctor_specialization.add(specialist[0])
            doctors_names = merged_df.loc[merged_df["Disease_x"] == disease, "Name"].values
            random_doctors = random.sample(list(doctors_names), min(10, len(doctors_names)))
            for name in random_doctors:  
                if pd.isna(name):
                    continue
                name = name.replace('\xa0', '') 
                lst.append((name, specialist[0]))
    return lst

@app.route('/api/doctors', methods=['POST'])
def get_doctors():
    words = set()
    for iteam in new_lst:
        lst = iteam['symptoms']
        for word in re.findall(r'\b\w+\b', lst):
            words.add(word)
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    
    filtered_doctors = predict_disease_and_doctor(symptoms, df)
    passage = re.findall(r'\b\w+\b', symptoms)
    passage = [word.lower() for word in passage if word.lower() not in stop_words]
    flag = True
    for i in passage:
        i = i.strip()
        if i in words:
            flag = False
    if flag:
        filtered_doctors = []
    return jsonify(filtered_doctors)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
