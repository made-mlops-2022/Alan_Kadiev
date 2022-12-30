import numpy as np
import pandas as pd
import pickle
import pathlib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.enities.feature_params import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan,
                                     strategy='most_frequent')),
            ('ohe', OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [('impute', SimpleImputer(missing_values=np.nan,
                                  strategy='mean')),
         ('scaler', StandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def make_features(transformer: ColumnTransformer,
                  df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target


def serialize_transformer(transformer: ColumnTransformer, output: str) -> str:
    absolute_path = str(pathlib.Path(__file__).parent.parent)
    absolute_path += output
    with open(absolute_path, "wb") as sf:
        pickle.dump(transformer, sf)
    return output


def deserialize_transformer(path: str) -> ColumnTransformer:
    absolute_path = str(pathlib.Path(__file__).parent.parent)
    absolute_path += path
    with open(absolute_path, "rb") as sf:
        transformer = pickle.load(sf)
    return transformer