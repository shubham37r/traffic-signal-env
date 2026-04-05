from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Traffic Signal Environment is running!"}
    