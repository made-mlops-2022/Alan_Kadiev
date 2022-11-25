import logging
import json
import sys
import pathlib
import click
import os

from ml_project.data import (make_data,
                             read_data,
                             download_data_from_s3
                             )
from ml_project.enities.train_pipeline_params import (TrainingPipelineParams,
                                                      read_training_pipeline_params,
                                                      )
from ml_project.features.build_features import build_transformer
from ml_project.models.model_train import (train_model,
                                           serialize_model,
                                           evaluate_model,
                                           )

from ml_project.models.model_train import create_inference_pipeline
from ml_project.models.model_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params: TrainingPipelineParams):
    downloading_params = training_pipeline_params.downloading_params

    if downloading_params:
        for path in downloading_params.paths:
            parent_path = str(pathlib.Path(__file__).parent.parent)
            absolute_output = parent_path + downloading_params.output_folder

            os.makedirs(absolute_output, exist_ok=True)

            download_data_from_s3(downloading_params.s3_bucket,
                                  path,
                                  absolute_output + "/" + path,
                                  )

    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Start train pipeline with params: {training_pipeline_params.train_params}")

    logger.info(f"Data shape: {data.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)

    train_features, val_df, train_target, val_target = make_data(data, training_pipeline_params, transformer)

    logger.info(f"Train_features shape: {train_features.shape}")
    logger.info(f"Val_df shape: {val_df.shape}")

    model = train_model(train_features,
                        train_target,
                        training_pipeline_params.train_params
                        )

    inference_pipeline = create_inference_pipeline(model, transformer)

    predicts = predict_model(inference_pipeline, val_df)

    metrics = evaluate_model(predicts, val_target)

    parent_path = str(pathlib.Path(__file__).parent.parent)
    absolute_metric_path = parent_path + training_pipeline_params.metric_path
    with open(absolute_metric_path, "w") as metric_file:

        json.dump(metrics, metric_file)

    logger.info(f"Model metrics: {metrics}")

    path_to_model = serialize_model(inference_pipeline,
                                    training_pipeline_params.output_model_path
                                    )

    logger.info("End train pipeline")
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()