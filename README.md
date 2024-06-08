# MedNavy
MedNavy is a cutting-edge healthcare platform developed to streamline patient care by intelligently matching patients with specialized doctors. By harnessing the power of advanced machine learning and modern web technologies, MedNavy provides users with personalized healthcare insights and doctor recommendations, significantly optimizing the process of finding the right medical specialist.

## Project Overview
MedNavy focuses on transforming patient inputs into actionable healthcare recommendations. Here's how it works:

- Symptom Input: Users start by entering their symptoms in the provided input area on the MedNavy platform.
- Disease Prediction: A custom-trained transformer model processes these symptoms to predict a list of the most probable diseases. This model treats the symptom description as a text classification problem.
- Doctor Recommendation: For each predicted disease, MedNavy provides details of specialized doctors available for consultation. This process is designed to guide patients efficiently to the right specialists, optimizing healthcare resource allocation.
- High Accuracy: Our transformer model achieves a 94% accuracy rate in predicting diseases and recommending doctors. This high level of accuracy is maintained through the use of standard interference techniques during model testing.
  
## Features
- Custom Transformer Model: A sophisticated machine learning model trained to predict diseases based on user-reported symptoms.
- High Accuracy: Delivers a 94% accuracy rate in disease prediction and doctor recommendations.
- Integrated System: Combines the strengths of Django, Flask, and React to provide a seamless user experience.
- Dynamic User Interface: Utilizes HTML, CSS, and JavaScript to create an interactive and responsive frontend.

## Technology Stack
- Frontend:

  -- HTML, CSS, and JavaScript: Forms the foundation of the user interface, providing a robust and dynamic front-end experience.
  
- Backend:

  -- Django: Manages user data and handles requests efficiently, forming the backbone of the server-side operations.
  -- Flask: Serves the machine learning model and processes predictions, acting as the bridge between the frontend and the ML model.
- Natural Lnaguage Processing: \[Custom Transformer Model] A specialized model designed for high-accuracy disease prediction based on text classification of symptoms.
