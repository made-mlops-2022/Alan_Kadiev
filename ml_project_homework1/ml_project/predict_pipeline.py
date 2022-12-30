import logging
import pathlib
import click

from data.make_dataset import read_data
from enities.predict_pipeline_params import (PredictPipelineParams,
                                             read_predict_pipeline_params,
                                             )
from features.build_features import make_features
from features.build_features import deserialize_transformer
from ml_project.models.model_predict import (predict_model,
                                             make_prediction,
                                             deserialize_model)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(levelname)s\t%(message)s",
                    )
logger = logging.getLogger(__name__)


def predict_pipeline(config_path: str):
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> str:
    logger.info(f"Start predict pipeline with model from {predict_pipeline_params.model_path}")

    test_data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"Data shape: {test_data.shape}")

    model = deserialize_model(predict_pipeline_params.model_path)

    logger.info(f"Model is {model.__class__}")

    prediction = predict_model(model,
                               test_data
                               )

    logger.info(f"Prediction shape: {prediction.shape}")
    path_to_prediction = make_prediction(prediction,
                                         predict_pipeline_params.prediction_path
                                         )

    logger.info(f"Prediction was recorded to {path_to_prediction}")

    logger.info("End predict pipeline")

    return path_to_prediction


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()