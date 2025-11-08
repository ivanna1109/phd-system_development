import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback, EarlyStopping
from model import create_nn_model , create_rf_model, create_xgb_model
from xai_metrics import log_shap_analysis, log_lime_analysis

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
        "--test_file",
        type=str,
        default='/home/ivana-milutinovic/Documents/Doktorske/Prva godina/RazvojSistema/gitHub/phd-system_development/data_processing/prepared_data/test_set.csv',
        help="Put do fajla sa podacima za finalnu evaluaciju (Test set)."
    )
    parser.add_argument(
        "--dvc_version",
        type=str,
        default='v1.2_clean',
        help="Verzija podataka koja je korišćena (DVC tag) za audit trail."
    )
    
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default='NN_Diabetes_Model', 
        help="Ime pod kojim ce model biti registrovan u MLflow Model Registru."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default='DNN_New_Run',
        help="Ime MLflow run-a za ovu konkretnu obuku (koristi se za razlikovanje unutar eksperimenta)."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default='DNN',
        choices=['DNN', 'RF', 'XGB'], 
        help="Tip modela za treniranje: DNN, RF (Random Forest) ili XGB (XGBoost)."
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
                new_name = f'val_{name}'
            else:
                # Metrike bez 'val_' prefiksa logujemo kao 'epoch_train_...'
                # Npr. loss postaje epoch_train_loss
                new_name = f'train_{name}'
                
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
        test_df = pd.read_csv(args.test_file) 
        print(f"Učitani podaci. Trening shape: {train_df.shape} | Validacija: {val_df.shape} | Test: {test_df.shape}")
    except FileNotFoundError as e:
        print(f"Greška pri učitavanju fajla: {e}. Proveri putanje!")
        return 

    X_train = train_df.drop('Diabetes_binary', axis=1)
    y_train = train_df['Diabetes_binary']
    X_val = val_df.drop('Diabetes_binary', axis=1)
    y_val = val_df['Diabetes_binary']
    X_test = test_df.drop('Diabetes_binary', axis=1)
    y_test = test_df['Diabetes_binary']

    INPUT_SHAPE = X_train.shape[1] 


    if args.model_type == 'DNN':
        model_params = {
            "epochs": 50, "batch_size": 32, "learning_rate": 0.0005, 
            "layer_1_units": 64, "layer_2_units": 32, "model_name": "NN_Diabetes_Model"
        }
        # Skaliranje za DNN
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train)
        X_val_final = scaler.transform(X_val)
        X_test_final = scaler.transform(X_test)
        model_to_train = create_nn_model(INPUT_SHAPE, model_params['layer_1_units'], model_params['layer_2_units'], model_params['learning_rate'])
        
    elif args.model_type == 'RF':
        model_params = {
            "n_estimators": 150, "max_depth": 8, "random_state": 42, "model_name": "RF_Diabetes_Model"
        }
        # RF i XGBoost ne zahtevaju skaliranje
        X_train_final = X_train 
        X_val_final = X_val
        X_test_final = X_test
        model_to_train = create_rf_model(model_params['n_estimators'], model_params['max_depth'], model_params['random_state'])
        
    elif args.model_type == 'XGB':
        model_params = {
            "n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": 42, "model_name": "XGB_Diabetes_Model"
        }
        X_train_final = X_train
        X_val_final = X_val
        X_test_final = X_test
        model_to_train = create_xgb_model(model_params['n_estimators'], model_params['learning_rate'], model_params['max_depth'], model_params['random_state'])
        
    else:
        raise ValueError(f"Nepodrzan tip modela: {args.model_type}")

    # --- 2. TRENIRANJE I LOGOVANJE ---
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

            test_dataset = mlflow.data.from_pandas(
            df=test_df,
            source=args.test_file,
            name="test_set"
            )
        
        
        # Logovanje dataset-a u MLflow Run
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")
            mlflow.log_input(test_dataset, context="evaluation")
        
        except ImportError:
            print("Upozorenje: Biblioteka mlflow.data nije pronađena. Logovanje dataset-a je preskočeno.")
        
        mlflow.log_params(model_params) # Loguje parametre specifične za odabrani model
        mlflow.log_param("model_type", args.model_type)
        mlflow.set_tag("dataset_dvc_version", args.dvc_version)
        
        # LOGIKA TRENINGA I EVALUACIJE ZA RAZLIČITE TIPOVE MODELA
        
        if args.model_type == 'DNN':
            # DNN Trening
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
            custom_logger = MLflowEpochLogger()
            callbacks_list = [custom_logger, early_stopping]
            
            history = model_to_train.fit(
                X_train_final, y_train,
                epochs=model_params['epochs'],
                batch_size=model_params['batch_size'],
                validation_data=(X_val_final, y_val),
                verbose=0,
                callbacks=callbacks_list 
            )
            # Evaluacija (DNN koristi evaluate)
            loss, acc, prec, rec = model_to_train.evaluate(X_test_final, y_test, verbose=0)
            
            # Logovanje modela za DNN (TensorFlow/Keras)
            mlflow.tensorflow.log_model(model_to_train, name="model_artifact", 
                                        registered_model_name=model_params['model_name'])
            
        else:
            # RF / XGBoost Trening (Sklearn stil)
            model_to_train.fit(X_train_final, y_train)
            
            # Evaluacija (Sklearn koristi predict/predict_proba)
            y_pred = model_to_train.predict(X_test_final)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            # Logovanje modela za Sklearn-bazirane modele
            mlflow.sklearn.log_model(model_to_train, artifact_path="model_artifact", 
                                     registered_model_name=model_params['model_name'])
            
            # Logujemo metrike za RF/XGBoost direktno (nema metrika po epohi)
            mlflow.log_metric("test_accuracy", acc)

        # 3. KREIRANJE FINALNIH METRIKA (Za sve modele)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

        final_metrics = {
            "final_accuracy": acc,
            "final_f1_score": f1,
            "final_recall": rec,
            "final_precision": prec,
        }
        mlflow.log_metrics(final_metrics) 

        plot_path = log_shap_analysis(
            model=model_to_train, 
            X_test_scaled=X_test_final, 
            X_test_df=X_test,
            model_type=args.model_type, 
            feature_names=X_test.columns.tolist() 
        )
        mlflow.log_artifact(plot_path)
        print(f"SHAP Summary Plot logovan u MLflow artefakte kao: {plot_path}")

        lime_path = log_lime_analysis(
            model=model_to_train, 
        X_test_df=X_test, 
        X_test_scaled=X_test_final, 
        model_type=args.model_type, 
        feature_names=X_test.columns.tolist() 
        )

        mlflow.log_artifact(lime_path)
        print(f"LIME objašnjenje za uzorak logovano kao: {lime_path}")

        print("-------------------------------------------------------")
        print(f"Model završen: {args.model_type}")
        print(f"Model registrovan kao: {model_params['model_name']}")
        print(f"Klinički Recall (Senzitivnost): {rec:.4f}")
        print("-------------------------------------------------------")

if __name__ == "__main__":
    main()