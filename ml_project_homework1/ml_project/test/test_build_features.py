import pytest
from typing import List

from ml_project.features.build_features import *
from ml_project.enities import FeatureParams


@pytest.fixture
def feature_params(categorical_features: List[str],
                   features_to_drop: List[str],
                   numerical_features: List[str],
                   target_col: str) -> FeatureParams:
    params = FeatureParams(categorical_features=categorical_features,
                           numerical_features=numerical_features,
                           features_to_drop=features_to_drop,
                           target_col=target_col,
                           )
    return params


def test_make_features(df: pd.DataFrame,
                       feature_params: FeatureParams):
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    features = make_features(transformer, df)
    assert not pd.isnull(features).any().any()


def test_extract_target(df: pd.DataFrame,
                        feature_params: FeatureParams):
    target = extract_target(df, feature_params)
    for i in range(len(target.to_numpy())):
        assert target[i] == df[feature_params.target_col][i]