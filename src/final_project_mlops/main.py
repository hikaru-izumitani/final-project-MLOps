import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from sqlalchemy import String, create_engine, JSON, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

app = FastAPI()

#https://docs.sqlalchemy.org/en/21/orm/quickstart.html

class Base(DeclarativeBase):
    pass

class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    features: Mapped[list[float]] = mapped_column(JSON)
    prediction_label: Mapped[str] = mapped_column(String(10))

engine = create_engine("sqlite:///predictions.db", echo=True)#, connect_args={"check_some_thread": False})
Base.metadata.create_all(engine)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
model = mlflow.sklearn.load_model("runs:/d6711e3a650e4b5d88a06c3843757bcc/model")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def post_predict(data: list[float]):
    prediction = model.predict([data])
    pred_label = str(int(prediction[0]))
    with Session(engine) as session:
        stmt = Model(features = data, prediction_label=pred_label)
        session.add(stmt)
        session.commit()
        session.refresh(stmt)
        record_id = stmt.id
    return {"prediction_id": record_id, "prediction": int(prediction[0])}

@app.get("/predict/{id}")
def get_predict_id(id: int):
    with Session(engine) as session:
        stmt = select(Model).where(Model.id == id)
        output = session.scalar(stmt)
        if output is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return {"prediction_id": id, "features": output.features, "prediction_label": output.prediction_label, }
