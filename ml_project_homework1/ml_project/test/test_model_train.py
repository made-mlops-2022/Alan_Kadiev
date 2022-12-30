import pytest
import pickle
import pathlib
import pandas as pd

from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier

from ml_project.enities import TrainingParams
from ml_project.models.model_train import train_model, serialize_model


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series], train_params: TrainingParams):
    dataset, target = features_and_target

    model = train_model(dataset, target, train_params=train_params)
    assert isinstance(model, RandomForestClassifier)


def test_serialize_model():
    expected_output = "/models/model.pkl"
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output

    absolute_path = str(pathlib.Path(__file__).parent.parent.parent)
    absolute_path += real_output
    with open(absolute_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)