import os
import pathlib
import click
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("Predict")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


@click.command("split")
@click.argument("input-dir")
@click.argument("output-dir")
def split(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    data_train, data_val, target_train, target_val = train_test_split(data, target, test_size=0.1, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    data_train.to_csv(os.path.join(output_dir, "data_train.csv"))
    target_train.to_csv(os.path.join(output_dir, "target_train.csv"))
    data_val.to_csv(os.path.join(output_dir, "data_val.csv"))
    target_val.to_csv(os.path.join(output_dir, "target_val.csv"))
    logger.warning('Data split task finished')


if __name__ == "__main__":
    split()