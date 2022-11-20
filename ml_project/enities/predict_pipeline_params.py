import yaml
import pathlib
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    transformer_path: str
    prediction_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    parent_path = str(pathlib.Path(__file__).parent.parent)
    absolute_path = parent_path + path
    with open(absolute_path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))