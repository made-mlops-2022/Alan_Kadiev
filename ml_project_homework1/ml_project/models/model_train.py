import pickle
import pathlib
import json
import numpy as np
import pandas as pd
from typing import Dict, Union

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from ml_project.enities.train_params import TrainingParams

SklearnRegressionModel = Union[RandomForestClassifier, GradientBoostingClassifier]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: TrainingParams) -> SklearnRegressionModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=train_params.model_RFC_params.n_estimators,
                                       random_state=train_params.model_RFC_params.random_state,
                                       max_depth=train_params.model_RFC_params.max_depth
                                       )
    elif train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(n_estimators=train_params.model_GBC_params.n_estimators,
                                           random_state=train_params.model_GBC_params.random_state,
                                           max_depth=train_params.model_GBC_params.max_depth
                                           )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def evaluate_model(predicts: np.ndarray,
                   target: pd.Series) -> Dict[str, float]:
    return {"accuracy": round(accuracy_score(target, predicts), 4),
            "precision": round(precision_score(target, predicts, average='macro'), 4),
            "recall": round(recall_score(target, predicts, average='macro'), 4),
            "f_1": round(f1_score(target, predicts, average='macro'), 4),
            }


def create_inference_pipeline(model: SklearnRegressionModel,
                              transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    absolute_path = str(pathlib.Path(__file__).parent.parent.parent)
    absolute_path += output
    with open(absolute_path, "wb") as f:
        pickle.dump(model, f)
    return output


def write_metrics(model_metrics: Dict[str, float], path: str):
    absolute_path = str(pathlib.Path(__file__).parent.parent.parent)
    absolute_path += path
    with open(absolute_path, "w") as sf:
        json.dump(model_metrics, sf)
    return path