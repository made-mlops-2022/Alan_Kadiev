import os
import click
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("model_train")
@click.argument("input-dir")
@click.argument("output-dir")
def train_model(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data_train.csv"), index_col=None)
    target = pd.read_csv(os.path.join(input_dir, "target_train.csv"), index_col=None)

    data_col = data.columns.tolist()[-30:]
    new_data = pd.DataFrame(data[data_col], columns=data_col)
    new_target = target['target']

    model = LogisticRegression(random_state=13, max_iter=100,)
    model.fit(new_data, new_target)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as handle:
        pickle.dump(model, handle)


if __name__ == "__main__":
    train_model()