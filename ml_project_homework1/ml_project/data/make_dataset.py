import pathlib

import boto3
import numpy as np
import pandas as pd
# import mlflow

from typing import Tuple, Union
from boto3 import client
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from ml_project.enities import SplittingParams
from ml_project.features import make_features, extract_target, build_transformer


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> None:
    s3 = boto3.client("s3")
    s3.download_file(s3_bucket, s3_path, output)


def read_data(path: str) -> pd.DataFrame:
    parent_path = str(pathlib.Path(__file__).parent.parent.parent)
    absolute_path = parent_path + path
    data = pd.read_csv(absolute_path)
    return data


def split_train_val_data(data: pd.DataFrame,
                         params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(data,
                                            test_size=params.val_size,
                                            random_state=params.random_state,
                                            shuffle=params.shuffle)
    return train_data, val_data


def make_data(data: pd.DataFrame,
              training_pipeline_params,
              transformer: ColumnTransformer) -> Tuple[Union[pd.DataFrame, np.ndarray],
                                                       Union[pd.DataFrame, np.ndarray],
                                                       Union[pd.Series, np.ndarray],
                                                       Union[pd.Series, np.ndarray],
]:


    train_df, val_df = split_train_val_data(data,
                                            training_pipeline_params.splitting_params
                                            )

    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(columns=training_pipeline_params.feature_params.target_col)

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    val_df = val_df.drop(columns=training_pipeline_params.feature_params.target_col)

    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    return train_features, val_df, train_target, val_target