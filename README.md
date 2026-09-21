# Intrusion Detection Platform

A real-time intrusion and anomaly detection platform built with FastAPI, Kafka, Faust, and Isolation Forest. 
This initial project is a minimal, Dockerized FastAPI service.

## Local Development (venv)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Docker

```powershell
docker build -t intrusion-detection-platform .
docker run -p 8000:8000 intrusion-detection-platform
```
