import pandas as pd
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
from ml_loader import get_model_and_scaler 

from django.shortcuts import render
from prediction_api import predict_utils
from ml_loader import get_model_and_scaler
from prediction_api import xai_utils as xai

def index(request):
    return render(request, 'index.html')


# Lista feature-a PIMA skupa (redosled MORA biti identičan kao u treningu!)
FEATURE_NAMES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 
                 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'PhysHlth', 'DiffWalk', 
                 'Sex', 'GenHlth_2.0', 'GenHlth_3.0', 'GenHlth_4.0', 'GenHlth_5.0', 'Age_2.0', 'Age_3.0', 'Age_4.0', 'Age_5.0', 
                 'Age_6.0', 'Age_7.0', 'Age_8.0', 'Age_9.0', 'Age_10.0', 'Age_11.0', 'Age_12.0', 'Age_13.0', 'Education_2.0',
                   'Education_3.0', 'Education_4.0', 'Education_5.0', 'Education_6.0', 'Income_2.0', 'Income_3.0', 'Income_4.0', 
                   'Income_5.0', 'Income_6.0', 'Income_7.0', 'Income_8.0']
CLASS_NAMES = ['Bez dijabetesa', 'Dijabetes']
REGISTERED_MODEL_NAMES = ["NN_Diabetes_Model", "RF_Diabetes_Model", "XGB_Diabetes_Model"]

# === INICIJALIZACIJA LIME EXPLAINER-a NA STARTU SERVERA ===
xai.initialize_lime_explainers(REGISTERED_MODEL_NAMES)

def expand_categorical_features(raw_data):
    """
    Pretvara originalni JSON sa sirovim podacima u format
    sa 45 One-Hot kodiranih kolona (kao u FEATURE_NAMES).
    """
    expanded = {}
    
    # Kategorijske mape (prema trening skupu)
    categorical_features = {
        'GenHlth': [2.0, 3.0, 4.0, 5.0],
        'Age': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        'Education': [2.0, 3.0, 4.0, 5.0, 6.0],
        'Income': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }
    
    # Prolazimo kroz sve feature-e koje model očekuje
    for feature in FEATURE_NAMES:
        # Ako je feature npr. Age_10.0
        if "_" in feature and any(feature.startswith(cat + "_") for cat in categorical_features):
            base, val = feature.split("_")
            val = float(val)
            if base in raw_data and raw_data[base] == val:
                expanded[feature] = 1.0
            else:
                expanded[feature] = 0.0
        else:
            # Direktna numerička vrednost
            expanded[feature] = float(raw_data.get(feature, 0.0))
    
    return expanded


@api_view(['POST'])
def predict_diagnosis(request):
    data = request.data
    models_list = data.get("models", ["DNN"])
    raw_data = data.get("data", data)

    data_expanded = expand_categorical_features(raw_data)

    # 2. Napravi DataFrame sa tačnim imenima kolona (isto kao u treningu!)
    input_df = pd.DataFrame([[data_expanded[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    results = {}
    for model_name in models_list:
        model, scaler = get_model_and_scaler(model_name)
    
        if model_name in ["NN_Diabetes_Model"]:
        # Skaliranje samo za DNN
            input_array_scaled = scaler.transform(input_df)
            pred_idx, confidence, probs = predict_utils.predict_keras_model(model, input_array_scaled)
        else:
        # Tree modeli ne skaliraju podatke
            pred_idx, confidence, probs = predict_utils.predict_model(model, input_df)

        lime_data = xai.generate_api_lime_explanation(
            sample_df=input_df, 
            model=model,
            scaler=scaler,
            model_type=model_name,
        )
        #print(f"Lime data:{lime_data}")
        results[model_name] = {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": f"{confidence:.4f}",
            "probabilities": {cls: f"{prob:.4f}" for cls, prob in zip(CLASS_NAMES, probs)},
            "lime_explanation": lime_data
        }

    return Response({
        "status": "success",
        "results": results,
        "message": "Predikcija dijagnoze je uspešno izvršena."
    }, status=status.HTTP_200_OK)
