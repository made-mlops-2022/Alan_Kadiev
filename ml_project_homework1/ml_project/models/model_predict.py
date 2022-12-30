import pickle
import pathlib
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

SklearnRegressionModel = Union[RandomForestClassifier, GradientBoostingClassifier]


def predict_model(model: Pipeline,
                  dataset: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(dataset)
    return predicts


def make_prediction(prediction: np.ndarray,
                    output: str) -> str:
    absolute_path = str(pathlib.Path(__file__).parent.parent)
    absolute_path += output
    with open(output, 'w') as sf:
        sf.write("id,prediction\n")
        for i in range(prediction.size):
            sf.write(f"{i},{prediction[i]}\n")
    return output


def deserialize_model(path: str) -> Pipeline:
    absolute_path = str(pathlib.Path(__file__).parent.parent)
    absolute_path += path
    with open(absolute_path, "rb") as sf:
        model = pickle.load(sf)
    return model