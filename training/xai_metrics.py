import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from functools import partial
import numpy as np

def log_shap_analysis(model, X_test_scaled, X_test_df, model_type, feature_names):
    """Izracunava i loguje SHAP Summary Plot kao artefakt."""
    
    # 1. Odredjivanje Explainer-a na osnovu tipa modela
    if model_type == 'DNN':
        background_data = X_test_scaled[:500] 
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(X_test_scaled)
        shap_values = shap_values[0]
        
    else:
        # Tree Explainer za RF i XGBoost
        explainer = shap.TreeExplainer(model, data=X_test_df.values) 
        shap_values = explainer.shap_values(X_test_df)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
            shap_values, 
            X_test_df, 
            feature_names=feature_names,
            show=False 
        ) 
    
    # 3. Cuvanje Plot-a kao artefakta
    full_path = '/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/training/xai_plots'
    plot_path = f"{full_path}/shap_0.44_summary_plot{model_type}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close() # Zatvori figuru da oslobodis memoriju
    
    return plot_path

def log_lime_analysis(model, X_test_df, X_test_scaled, model_type, feature_names, class_names=['Bez dijabetesa', 'Dijabetes']):
    """Generiše i loguje LIME objašnjenje za prvi uzorak Test Seta."""

    if model_type == 'DNN':
        # DNN model daje direktno verovatnoće
        def dnn_predict_proba(data):
            # 1. Dobijanje predikcije za klasu 1 (verovatnoća, oblika (N, 1))
            proba_class_1 = model.predict(data)
            
            # 2. Izračunavanje verovatnoće za klasu 0 (1 - P(klasa 1))
            proba_class_0 = 1 - proba_class_1
            
            # 3. Spajanje u format (N, 2) koji LIME očekuje
            return np.concatenate((proba_class_0, proba_class_1), axis=1)

        predict_fn = dnn_predict_proba # Koristimo prilagođenu funkciju
    else:
        # Sklearn modeli (RF, XGB) koriste predict_proba
        predict_fn = model.predict_proba

    # U zavisnosti od modela, LIMEu trebaju razliciti podaci
    if model_type == 'DNN':
        # Za DNN: LIME u pozadini radi predikcije, pa mora znati da su podaci skalirani
        data_to_explain = X_test_scaled # Prosleđujemo skalirane
    else:
        # Za RF/XGB: LIME radi na ne-skaliranim podacima
        data_to_explain = X_test_df.values # Prosleđujemo vrednosti Data Frame-a

    # 2. Inicijalizacija LIME Tabular Explainer-a
    # Koristimo neskalirane vrednosti za trening (za statistiku)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test_df.values, # LIME uvek preferira neskalirane podatke za statistiku
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

    # 3. Izbor uzorka za objašnjenje (npr., prvi uzorak iz Test Seta)
    sample_to_explain = data_to_explain[0]
    
    # 4. Generisanje LIME objašnjenja
    explanation = explainer.explain_instance(
        data_row=sample_to_explain,
        predict_fn=predict_fn,
        num_features=10 # Objasni 10 najvažnijih atributa
    )

    fig = explanation.as_pyplot_figure(label=1) # label=1 za klasu 'Dijabetes'
    
    # 6. Cuvanje Plot-a kao PNG artefakta
    full_path = '/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/training/xai_plots'
    lime_path = f"{full_path}/lime_0.2.0.1_explanation_{model_type}.png"
    fig.savefig(lime_path, bbox_inches='tight', dpi=300)
    plt.close(fig) 
    
    return lime_path