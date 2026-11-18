import mlflow.sklearn
from fastapi import FastAPI

app = FastAPI()

mlflow.set_tracking_uri("http://127.0.0.1:8080")
model = mlflow.sklearn.load_model("runs:/d6711e3a650e4b5d88a06c3843757bcc/model")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: list[float]):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}