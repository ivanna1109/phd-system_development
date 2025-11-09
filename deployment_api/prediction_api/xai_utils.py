import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import os
from prediction_api.predict_utils import predict_model

# === Globalne varijable ===
LIME_EXPLAINER = {} 
CLASS_NAMES = ['Bez dijabetesa', 'Dijabetes']
TARGET_COLUMN = 'Diabetes_binary'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIME_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'lime_train_set.csv') 

def load_and_prepare_training_data():
    """Učitava train_data.csv, uklanja target kolonu i vraća numpy niz i imena feature-a."""
    try:
        # Učitavanje celog skupa
        df_train = pd.read_csv(LIME_TRAIN_DATA_PATH)
        print(f"INFO: Učitan CSV fajl za LIME (Originalni shape: {df_train.shape}).")
        
        # Uklanjanje target kolone (LIME-u trebaju samo feature-i)
        if TARGET_COLUMN in df_train.columns:
            df_features = df_train.drop(columns=[TARGET_COLUMN])
        else:
            df_features = df_train # Ako fajl već sadrži samo feature
            
        return df_features.values, df_features.columns.tolist()
    
    except FileNotFoundError:
        print(f"FATALNA GREŠKA: Nije pronađen LIME Training Data na putanji: {LIME_TRAIN_DATA_PATH}")
        return None, None
    except Exception as e:
        print(f"GREŠKA pri obradi Training Data za LIME: {e}")
        return None, None


def initialize_lime_explainers(model_names):
    """
    Inicijalizuje LIME Explainer koristeći lokalno sačuvane podatke.
    """
    global LIME_EXPLAINER
    
    TRAINING_DATA_NP, FEATURE_NAMES = load_and_prepare_training_data()

    if TRAINING_DATA_NP is None:
        return

    # Inicijalizacija LIME Explainer-a
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=TRAINING_DATA_NP,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        mode='classification',
        random_state=42,
        # Onemogućavanje diskretizacije je ključno za izbegavanje pickle/lambda grešaka
        discretize_continuous=False 
    )
    
    # Čuvamo istu instancu explainer-a za sve modele
    for name in model_names:
        LIME_EXPLAINER[name] = explainer

    print(f"INFO: LIME Explainer uspešno inicijalizovan za modele: {model_names}")


def generate_api_lime_explanation(sample_df, model, scaler, model_type):
    """Generiše LIME objašnjenje za jednu instancu (API poziv)."""
    
    explainer = LIME_EXPLAINER.get(model_type, None)
    if explainer is None:
        print("Explainer je null")
        return []

    # === 1. Definisanje predict_fn (Wrapper za Skaliranje/Predikciju) ===
    if model_type in ['NN_Diabetes_Model']:
        # Wrapper za DNN koji skalira unutar sebe
        def predict_fn(data):
            # LIME generiše ne-skalirane podatke, mi ih moramo skalirati pre predikcije
            data_scaled = scaler.transform(data) 
            proba_class_1 = model.predict(data_scaled)
            proba_class_0 = 1 - proba_class_1
            return np.concatenate((proba_class_0, proba_class_1), axis=1)
    else:
        # Sklearn modeli (RF, XGB)
        def predict_fn(data_array): 
            data_df = pd.DataFrame(data_array, columns=explainer.feature_names)
            
            all_probs = []
            for index, row in data_df.iterrows():
                # Prosleđujemo svaki red kao DataFrame (1, N_FEATURES)
                _, _, probs_single = predict_model(model, pd.DataFrame([row], columns=data_df.columns))
                
                all_probs.append(probs_single) 
                
            return np.array(all_probs)

    sample_to_explain = sample_df.values[0]

    # === 2. Generisanje LIME objašnjenja ===
    explanation = explainer.explain_instance(
        data_row=sample_to_explain,
        predict_fn=predict_fn,
        num_features=8
    )

    # Vraćamo objašnjenje za klasu 1 (Dijabetes) kao lista torki
    return explanation.as_list(label=1)