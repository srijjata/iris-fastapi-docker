from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import joblib
import os

class MyModel:
    def __init__(self):
        print("MyModel created")
        self.model = RandomForestClassifier()

    def train_model(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        features_name = iris.feature_names
        target_names = iris.target_names

        #train-test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42, stratify=y)
        self.model.fit(X_train,y_train)

    def save_model(self):  
        #Save the trained model
        with open("../model/iris/iris_class_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float   


class MultipleIrisFeatures(BaseModel):
    features: list[IrisFeatures]
