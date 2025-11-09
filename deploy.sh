#!/bin/bash


CONTAINER_NAME="diabetes_service"
IMAGE_NAME="diabetes-api:latest"
API_PATH="./deployment_api"
API_PORT="8000"


echo "--- 1. PROVERA I ZAUSTAVLJANJE POSTOJEĆEG KONTEJNERA ---"
if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
    echo "Zaustavljanje i uklanjanje starog kontejnera: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
else
    echo "Kontejner $CONTAINER_NAME ne postoji, nastavljam..."
fi

echo -e "\n--- 2. IZGRADNJA (BUILD) DOCKER SLIKE ---"
docker build -t $IMAGE_NAME $API_PATH

if [ $? -ne 0 ]; then
    echo "GREŠKA: Izgradnja Docker slike je neuspešna. Proverite Dockerfile i logove."
    exit 1
fi

echo -e "\n--- 3. POKRETANJE NOVOG KONTEJNERA ---"
docker run -d --name $CONTAINER_NAME \
    -p $API_PORT:$API_PORT \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo -e "\n--- USPEH ---"
    echo "Django API servis je uspešno pokrenut."
    echo "Proverite logove sa: docker logs $CONTAINER_NAME"
    echo "API je dostupan na: http://localhost:$API_PORT/api/predict/"
else
    echo "GREŠKA: Pokretanje kontejnera je neuspešno. Proverite Docker logove."
fi

exit 0