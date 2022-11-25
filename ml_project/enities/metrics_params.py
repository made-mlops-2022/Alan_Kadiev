from dataclasses import dataclass, field


@dataclass()
class MetricParams:
    accuracy: bool = field(default=True)
    recall: bool = field(default=True)
    precision: bool = field(default=True)
    f_1: bool = field(default=True)
