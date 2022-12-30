import pytest
import pandas as pd

from ml_project.enities.split_params import SplittingParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.train_pipeline_params import TrainingPipelineParams
from ml_project.train_pipeline import run_train_pipeline


@pytest.fixture()
def train_pipeline_params(target_col,
                          categorical_features,
                          numerical_features,
                          features_to_drop,
                          train_params,
                          metric_params) -> TrainingPipelineParams:
    splitting = SplittingParams(random_state=42, val_size=0.2, shuffle=True)
    features = FeatureParams(categorical_features,
                             numerical_features,
                             features_to_drop,
                             target_col)
    return TrainingPipelineParams(input_data_path="/data/raw/train.csv",
                                  output_model_path="/models/model.pkl",
                                  metric_path="/models/metrics.json",
                                  splitting_params=splitting,
                                  feature_params=features,
                                  train_params=train_params,
                                  metric_params=metric_params)


def test_run_train_pipeline(mocker,
                            df: pd.DataFrame,
                            train_pipeline_params: TrainingPipelineParams):
    func = "ml_project.data.make_dataset.read_data"
    mocker.patch(func, return_value=df)
    model_path, metrics = run_train_pipeline(train_pipeline_params)
    assert model_path == "/models/model.pkl"
    assert "accuracy" in metrics.keys()
    assert "precision" in metrics.keys()
    assert "recall" in metrics.keys()
    assert "f_1" in metrics.keys()