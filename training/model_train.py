import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback, EarlyStopping
from mlflow.data.pandas_dataset import PandasDataset
# Pretpostavljamo da create_nn_model ostaje u model.py i uvozi se
from model import create_nn_model 

# --- 0. DEFINISANJE ARGUMENATA KOMANDNE LINIJE (ARGPARSE) ---
def parse_args():
    """Parsira argumente iz komandne linije za MLOps pipeline."""
    parser = argparse.ArgumentParser(description="MLflow MLOps Training Script for Diabetes NN")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default='klasifikacija_dijabetesa_nn',
        help="Ime MLflow eksperimenta (obično fiksno za jedan problem)."
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        default='/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/prepared_data/train_set.csv',
        help="Put do fajla sa podacima za trening (apsolutna ili relativna putanja)."
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default='/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/prepared_data/validation_set.csv',
        help="Put do fajla sa podacima za validaciju."
    )
    parser.add_argument(
        "--dvc_version",
        type=str,
        default='v_default_20251107',
        help="Verzija podataka koja je korišćena (DVC tag) za audit trail."
    )
    
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default='NN_Diabetes_Model', # Koristimo fiksno ime za registar da bismo dobijali verzije (v1, v2, v3...)
        help="Ime pod kojim ce model biti registrovan u MLflow Model Registru."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default='DNN_New_Run',
        help="Ime MLflow run-a za ovu konkretnu obuku (koristi se za razlikovanje unutar eksperimenta)."
    )
    return parser.parse_args()

class MLflowEpochLogger(Callback):
    """
    Custom Keras Callback za logovanje metrika obuke i validacije po epohi
    direktno u MLflow.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        for name, value in logs.items():
            if name.startswith('val_'):
                # Metrike sa 'val_' prefiksom logujemo kao 'epoch_val_...'
                # Npr. val_loss postaje epoch_val_loss
                new_name = f'epoch_{name}'
            else:
                # Metrike bez 'val_' prefiksa logujemo kao 'epoch_train_...'
                # Npr. loss postaje epoch_train_loss
                new_name = f'epoch_train_{name}'
                
            mlflow.log_metric(new_name, value, step=epoch)
        
        # Opcioni ispis u konzolu radi pracenja
        print(f"MLflow Log: Epoch {epoch+1}/{self.params['epochs']} - Train Loss: {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}")

def main():
    args = parse_args()
    mlflow.tensorflow.autolog(disable=True)
    mlflow.end_run()
    
    mlflow.set_experiment(args.experiment_name)
    
    try:
        train_df = pd.read_csv(args.train_file)
        val_df = pd.read_csv(args.validation_file)
        print(f"Učitani podaci. Trening: {args.train_file} | Validacija: {args.validation_file}")
    except FileNotFoundError as e:
        print(f"Greška pri učitavanju fajla: {e}. Proveri putanje!")
        return # Prekida skriptu ako fajl nije pronađen
    

    X_train = train_df.drop('Diabetes_binary', axis=1)
    y_train = train_df['Diabetes_binary']
    X_val = val_df.drop('Diabetes_binary', axis=1)
    y_val = val_df['Diabetes_binary']

    INPUT_SHAPE = X_train.shape[1] 


    params = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "layer_1_units": 64,
        "layer_2_units": 32
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    nn_model = create_nn_model(INPUT_SHAPE, params['layer_1_units'], 
                                 params['layer_2_units'], params['learning_rate'])


    with mlflow.start_run(run_name=args.run_name) as run:
        
        # 5a. Logovanje hiperparametara
        try:

        # Kreiranje objekta za Trening set
            train_dataset = mlflow.data.from_pandas(
            df=train_df,
            source=args.train_file,
            name="train_set"
            )
        # Kreiranje objekta za Validacioni set
            val_dataset = mlflow.data.from_pandas(
            df=val_df,
            source=args.validation_file,
            name="validation_set"
            )
        
        # Logovanje dataset-a u MLflow Run
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")
        
        # Opciono: Logovanje DVC verzije kao tag za dataset
            mlflow.set_tag("dataset_dvc_version", args.dvc_version)

        except ImportError:
            print("Upozorenje: Biblioteka mlflow.data nije pronađena. Logovanje dataset-a je preskočeno.")
        
        mlflow.log_params(params)
        mlflow.log_param("input_features", INPUT_SHAPE)
        mlflow.log_param("DVC_Data_Version", args.dvc_version) # Logovanje DVC verzije

        early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        mode='min', 
        restore_best_weights=True,
        verbose=1
        )
        # 5b. Treniranje
        custom_logger = MLflowEpochLogger()
        callbacks_list = [custom_logger, early_stopping]
    
        history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_val_scaled, y_val),
        verbose=0,
        callbacks=callbacks_list 
        )
        
        # 5c. Evaluacija
        loss, acc, prec, rec = nn_model.evaluate(X_val_scaled, y_val, verbose=0)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

        metrics = {
            "val_accuracy": acc,
            "val_f1_score": f1,
            "val_recall_1": rec,
            "val_precision_1": prec,
            "val_loss": loss
        }
        
        # 5d. Logovanje Metrika
        #mlflow.log_metrics(metrics)
        
        # 5e. Logovanje i registracija Modela
        mlflow.tensorflow.log_model(nn_model, 
                                    name="dnn_model",
                                    registered_model_name=args.registered_model_name)
        
        # --- IZLAZ ---
        print("-------------------------------------------------------")
        print(f"Eksperiment MLflow završen u '{args.experiment_name}'.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model registrovan kao: {args.registered_model_name} (Nova verzija)")
        print(f"Korišćena DVC verzija podataka: {args.dvc_version}")
        print(f"Klinički Recall (Senzitivnost): {metrics['val_recall_1']:.4f}")
        print("-------------------------------------------------------")

if __name__ == "__main__":
    main()