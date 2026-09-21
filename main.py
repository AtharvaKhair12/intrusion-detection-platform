from fastapi import FastAPI

app = FastAPI(title="Intrusion Detection Platform")

@app.get("/health")
def health_check():
    return {"status": "ok"}
