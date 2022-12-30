import yaml
import pathlib

from typing import Optional
from dataclasses import dataclass
from .download_params import DownloadParams
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from .metrics_params import MetricParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    metric_params: MetricParams

    downloading_params: Optional[DownloadParams] = None
    # use_mlflow: bool = True
    # mlflow_uri: str = "http://18.156.5.226/"
    # mlflow_experiment: str = "inference_demo"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    parent_path = str(pathlib.Path(__file__).parent.parent.parent)
    absolute_path = parent_path + path
    with open(absolute_path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))