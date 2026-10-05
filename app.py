from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import JSONResponce 
from fastapi.responses import StreamingResponse #both responses can be used
from fastapi.middleware.cors import CORSMiddleware
from sklearn.datasets import load_iris
import numpy as np

from pydantic import BaseModel
import logging #always use logs
import uvicorn
from pyngrok import ngrok
from fastapi.responses import StreamingResponse
from my_model import MyModel, IrisFeatures, MultipleIrisFeatures
import pickle

train_model = MyModel()
train_model.train_model()
train_model.save_model()

with open("../model/iris/iris_class_model.pkl", "rb") as f:
    model = pickle.load(f)

#API app
app = FastAPI(
    title="<...ML API...>",
    description="""This is a REST API that takes the IRIS file inputs 
    and returns predictions made by a selected machine learning model.""",
    version="0.1"
)

class_names ={
    0:"setosa",
    1:"versicolor",
    2:"virginica"
}

@app.get("/")
def root():
    return {
        "message":"Iris ML API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict_class(data: IrisFeatures):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    predicted_index = int(prediction.ravel()[0])

    return {"predicted_class": class_names[predicted_index]}


@app.post("/predict_batch")
def predict_batch(data: MultipleIrisFeatures):
    features = np.array([[feature.sepal_length, feature.sepal_width, feature.petal_length, feature.petal_width] for feature in data.features])

    predictions = model.predict(features)
    predicted_classes = [
    class_names[int(prediction)]
        for prediction in predictions
    ]

    return {"predicted_class": predicted_classes}