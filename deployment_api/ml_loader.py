import mlflow
from mlflow.tracking import MlflowClient
import os, pickle, tempfile

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_ARTIFACT_PATH = "model_artifact"
SCALER_ARTIFACT_NAME = "preprocessor_scaler.pkl"

# Keširanje po modelu
CACHED_MODELS = {}
CACHED_SCALERS = {}

class MLModelLoader:
    """Dinamičko učitavanje više modela iz MLflow-a."""
    
    def __init__(self, model_name=None):
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        else:
            raise EnvironmentError("MLFLOW_TRACKING_URI nije postavljen.")
        self.model_name = model_name
        self.client = MlflowClient()

    def load_production_model_and_scaler(self, model_name=None):
        """Učitava Production model i njegov skaler, kešira po model_name."""
        global CACHED_MODELS, CACHED_SCALERS

        model_name = model_name or self.model_name
        if model_name in CACHED_MODELS and model_name in CACHED_SCALERS:
            return CACHED_MODELS[model_name], CACHED_SCALERS[model_name]

        # Pronađi Production verziju
        production_version = self.client.get_latest_versions(model_name, stages=["Production"])
        if not production_version:
            raise ValueError(f"Nije pronađena Production verzija za model '{model_name}'")
        latest_version = production_version[0]
        run_id = latest_version.run_id

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logged_model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
                model = mlflow.pyfunc.load_model(logged_model_uri)

                if "NN" in model_name or "DNN" in model_name:
                    self.client.download_artifacts(run_id, SCALER_ARTIFACT_NAME, temp_dir)
                    full_scaler_path = os.path.join(temp_dir, SCALER_ARTIFACT_NAME)
                    with open(full_scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                else:
                    scaler = None  # Tree modeli nemaju skaler

        except Exception as e:
            raise RuntimeError(f"Neuspelo učitavanje modela/skalera: {e}")

        # Keširanje po imenu modela
        CACHED_MODELS[model_name] = model
        CACHED_SCALERS[model_name] = scaler
        return model, scaler


# Funkcija za lak pristup keširanim modelima
def get_model_and_scaler(model_name):
    global CACHED_MODELS, CACHED_SCALERS
    if model_name not in CACHED_MODELS or model_name not in CACHED_SCALERS:
        loader = MLModelLoader()
        return loader.load_production_model_and_scaler(model_name)
    return CACHED_MODELS[model_name], CACHED_SCALERS[model_name]
