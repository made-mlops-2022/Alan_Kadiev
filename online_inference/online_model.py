import os
import logging
from typing import List, Union, Optional
import pandas as pd
import pickle
import uvicorn
import gdown

from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_URL = "https://drive.google.com/file/d/1ZRoxFOganPtsEe0ijfSaw9tkm_T2m30C/view?usp=sharing"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('%(levelname)s: %(name)s - %(asctime)s - %(message)s'))
logger.addHandler(handler)

app = FastAPI()
model = None
scaler = None


class Request(BaseModel):
    data: List[conlist(Union[float, None], min_items=30, max_items=30)]
    features: List[str]


class Prediction(BaseModel):
    condition: int


@app.get("/")
def main():
    return {"message": "App for predictions heart diseases <3"}


@app.on_event("startup")
def startup():
    global model
    model_local_path = "model.pkl"
    logger.info(f" if model exists: {os.path.exists(model_local_path)}")

    if not os.path.exists(model_local_path):
        model_url = os.getenv("MODEL_URL")
        if model_url is None:
            model_url = DEFAULT_MODEL_URL
            logger.info("loading model from Google-Drive")
        gdown.download(model_url, output=model_local_path, fuzzy=True)

    with open(model_local_path, "rb") as file:
        model = pickle.load(file)
    logger.info("model upload")


@app.get("/health")
def health() -> int:
    return not (model is None)


@app.get("/predict/", response_model=List[Prediction])
def predict(request: Request):
    data = pd.DataFrame(request.data, columns=request.features)
    predictions = model.predict(data).tolist()
    return [Prediction(condition=int(lbl))
            for lbl in predictions
            ]


if __name__ == "__main__":
    uvicorn.run("online_model:app", host="0.0.0.0",
                port=os.getenv("PORT", 8080))