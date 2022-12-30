import pytest
import numpy as np
import pandas as pd
from typing import List, Tuple

from ml_project.enities import ModelRFCParams

from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.train_params import TrainingParams
from ml_project.enities.metrics_params import MetricParams
from ml_project.features.build_features import make_features, extract_target, build_transformer

from ml_project.enities.train_pipeline_params import TrainingPipelineParams


@pytest.fixture
def df() -> pd.DataFrame:
    np.random.seed(13)
    dataset = pd.DataFrame()
    dataset_size = 50
    dataset["cp"] = np.random.randint(0, 4, size=dataset_size)
    dataset["restecg"] = np.random.randint(0, 3, size=dataset_size)
    dataset["slope"] = np.random.randint(0, 3, size=dataset_size)
    dataset["ca"] = np.random.randint(0, 4, size=dataset_size)
    dataset["thal"] = np.random.randint(0, 3, size=dataset_size)
    dataset["sex"] = np.random.randint(0, 2, size=dataset_size)
    dataset["fbs"] = np.random.randint(0, 2, size=dataset_size)
    dataset["exang"] = np.random.randint(0, 2, size=dataset_size)
    dataset["age"] = np.random.randint(20, 90, size=dataset_size)
    dataset["trestbps"] = np.random.randint(60, 200, size=dataset_size)
    dataset["chol"] = np.random.randint(120, 600, size=dataset_size)
    dataset["thalach"] = np.random.randint(70, 200, size=dataset_size)
    dataset["oldpeak"] = np.round(np.random.uniform(0, 7, size=dataset_size), 1)
    dataset["condition"] = np.random.randint(0, 2, size=dataset_size)
    return dataset


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["sex",
            "cp",
            "fbs",
            'restecg',
            'exang',
            'slope',
            'ca',
            'thal'
            ]


@pytest.fixture
def numerical_features() -> List[str]:
    return ['oldpeak',
            'age',
            'chol',
            'thalach',
            'trestbps',
            ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["index"]


@pytest.fixture
def features(df: pd.DataFrame,
             categorical_features: List[str],
             numerical_features: List[str]) -> FeatureParams:
    return FeatureParams(categorical_features=categorical_features,
                         numerical_features=numerical_features,
                         features_to_drop=["index"],
                         target_col="condition",
                         )


@pytest.fixture()
def train_params() -> TrainingParams:
    return TrainingParams(model_type="RandomForestClassifier",
                          model_RFC_params=ModelRFCParams()
                          )


@pytest.fixture()
def metric_params() -> MetricParams:
    return MetricParams(accuracy=True,
                        precision=True,
                        recall=True,
                        f_1=True
                        )


@pytest.fixture
def features_and_target(df: pd.DataFrame,
                        features: FeatureParams) -> Tuple[pd.DataFrame, pd.Series]:

    target = extract_target(df, features)
    df.drop(columns=features.target_col)
    transformer = build_transformer(features)
    transformer.fit(df)
    dataset = make_features(transformer, df)
    return dataset, target