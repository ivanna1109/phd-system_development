#!/bin/bash

API_PORT="8001"

echo "--- POKRETANJE LOKALNOG DIABETES API SERVISA (PREKO GUNICORN-a) ---"
echo "PROVERITE da li MLflow server radi u drugom terminalu na http://127.0.0.1:5001"
echo "Servis će pokušati da učita model iz 'Production' faze."
echo "API je dostupan na: http://localhost:$API_PORT/api/predict/"
echo "------------------------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:./deployment_api

gunicorn --bind 0.0.0.0:$API_PORT deployment_api.diabetes_mlops.wsgi:application