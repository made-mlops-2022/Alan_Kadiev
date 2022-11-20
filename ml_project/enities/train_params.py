from typing import Optional
from dataclasses import dataclass, field

from .model_RFC_params import ModelRFCParams
from .model_GBC_params import ModelGBCParams


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    model_RFC_params: Optional[ModelRFCParams] = None
    model_GBC_params: Optional[ModelGBCParams] = None