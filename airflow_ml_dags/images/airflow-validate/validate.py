import os
import click
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command("validate")
@click.argument("input-dir")
@click.argument("model-dir")
@click.argument("output-dir")
def validate(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data_val.csv"), index_col=None)
    target = pd.read_csv(os.path.join(input_dir, "target_val.csv"), index_col=None)

    data_col = data.columns.tolist()[-30:]
    new_data = pd.DataFrame(data[data_col], columns=data_col)
    new_target = target['target']

    with open(os.path.join(model_dir, "model.pkl"), "rb") as handle:
        model = pickle.load(handle)

    prediction = model.predict(new_data)

    accuracy = accuracy_score(new_target, prediction)
    f1 = f1_score(new_target, prediction)

    metrics = {"accuracy": accuracy, "f1-score": f1}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle)


if __name__ == "__main__":
    validate()