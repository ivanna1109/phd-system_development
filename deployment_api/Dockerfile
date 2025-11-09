FROM python:3.10-slim

# WORKDIR je /app (gde ce biti manage.py, diabetes_mlops/ folder, itd.)
WORKDIR /app

# Kopiranje i instalacija zavisnosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiranje celog projekta u /app
COPY . /app/

EXPOSE 8000


# MLFLOW TRACKING URI: Obezbeđuje da se kontejner poveže na host mašinu
# Koristimo host.docker.internal umesto 172.17.0.1
ENV MLFLOW_TRACKING_URI="http://172.17.0.1:5001"

# Ova varijabla je neophodna da Docker kontejner pristupi hostu
ENV MLFLOW_DISABLE_HOST_HEADER_CHECK=True

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "diabetes_mlops.wsgi:application", "--chdir", "diabetes_mlops", "-w", "1", "--access-logfile", "-",\ 
    "--error-logfile", "-"]