import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import mlflow
from model import create_nn_model

# --- 1. POSTAVKE MLFLOW EKSPERIMENTA ---
EXPERIMENT_NAME = "klasifikacija_dijabetesa_nn"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2. UČITAVANJE PODATAKA ---
try:
    base_path = '/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/prepared_data/'
    # Pretpostavljamo da su setovi sačuvani u 'data/' folderu
    train_df = pd.read_csv(f'{base_path}train_set.csv')
    val_df = pd.read_csv(f'{base_path}validation_set.csv')
except FileNotFoundError:
    print("Greška: Nisu pronađeni 'train_set.csv' ili 'validation_set.csv'. Proveri DVC korak!")
    exit()

X_train = train_df.drop('Diabetes_binary', axis=1)
y_train = train_df['Diabetes_binary']
X_val = val_df.drop('Diabetes_binary', axis=1)
y_val = val_df['Diabetes_binary']

INPUT_SHAPE = X_train.shape[1] 

params = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "layer_1_units": 64,
    "layer_2_units": 32
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

nn_model = create_nn_model(INPUT_SHAPE, params['layer_1_units'], 
                             params['layer_2_units'], params['learning_rate'])


with mlflow.start_run(run_name="NN_Base_V1") as run:
    
    # Logovanje hiperparametara pre treninga
    mlflow.log_params(params)
    mlflow.log_param("input_features", INPUT_SHAPE)

    # 5a. Treniranje
    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_val_scaled, y_val),
        verbose=0 # Ispiši samo konačne rezultate
    )
    
    # 5b. Evaluacija (na Validacionom Setu)
    loss, acc, prec, rec = nn_model.evaluate(X_val_scaled, y_val, verbose=0)
    
    # 5c. Izračunavanje F1 
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

    metrics = {
        "val_accuracy": acc,
        "val_f1_score": f1,
        "val_recall_1": rec,
        "val_precision_1": prec,
        "val_loss": loss
    }
    
    # 5d. Logovanje Metrika
    mlflow.log_metrics(metrics)
    
    # 5e. Logovanje Modela (Keras/TensorFlow se loguje posebno)
    mlflow.tensorflow.log_model(nn_model, "dnn_model",
                                registered_model_name="NN_Diabetes_Model")
    
    # Logovanje verzije podataka
    mlflow.log_param("DVC_Data_Version", "v1.0_processed_20251106") 

    print("-------------------------------------------------------")
    print(f"Eksperiment MLflow NN V1 završen.")
    print(f"Model registrovan kao: NN_Diabetes_Model")
    print(f"Klinički Recall (Senzitivnost): {metrics['val_recall_1']:.4f}")
    print("-------------------------------------------------------")