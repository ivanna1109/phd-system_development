echo "--- POKRETANJE LOKALNOG MLFLOW SERVERA ---"
echo "MLflow Server hostuje bazu i artefakte na http://127.0.0.1:5001"
echo "------------------------------------------------------------------"

mlflow server \
    --backend-store-uri sqlite:///mlruns/mlruns.db \
    --artifacts-destination ./mlruns \
    --host 127.0.0.1 \
    --port 5001 \
    --serve-artifacts