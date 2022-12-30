import pytest
import pickle
import pathlib
import pandas as pd

from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from ml_project.enities.feature_params import FeatureParams
from ml_project.enities import TrainingParams
from ml_project.enities.split_params import SplittingParams
from ml_project.features.build_features import make_features, extract_target, build_transformer
from ml_project.models.model_predict import predict_model, make_prediction, deserialize_model
from sklearn.pipeline import Pipeline
from ml_project.models.model_train import (train_model,
                                           create_inference_pipeline,
                                           )
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@pytest.fixture
def pipeline_transformer(df: pd.DataFrame,
                         features: FeatureParams,
                         train_params: TrainingParams) -> Tuple[Pipeline, ColumnTransformer]:
    target = extract_target(df, features)
    df.drop(columns=features.target_col)

    transformer = build_transformer(features)
    transformer.fit(df)

    transformed_data = make_features(transformer, df)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(transformed_data, target)
    return create_inference_pipeline(model, transformer), transformer


def test_predict_model(df: pd.DataFrame,
                       features: FeatureParams,
                       pipeline_transformer: Tuple[Pipeline, ColumnTransformer]):
    model, transformer = pipeline_transformer

    target = extract_target(df, features)
    df.drop(columns=features.target_col)
    predict = predict_model(model, df)
    assert accuracy_score(target, predict) >= 0.8
    assert precision_score(target, predict, average='macro') >= 0.8
    assert predict.shape[0] == target.shape[0]