from dataclasses import dataclass, field


@dataclass()
class ModelRFCParams:
    n_estimators: int = field(default=50)
    random_state: int = field(default=13)
    max_depth: int = field(default=5)