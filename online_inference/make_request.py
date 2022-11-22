import pandas as pd
import requests
import logging
from _pytest import pathlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s'))
logger.addHandler(handler)


def main():
    test_file = "data/data.csv"
    url = "http://0.0.0.0:8080/predict/"

    path = pathlib.Path(__file__).parent.parent.joinpath(test_file)
    data = pd.read_csv(path, index_col=False).drop(columns="Unnamed: 0")
    logger.info(f"load data: shape - {data.shape}")

    features = list(data.columns)
    logger.info(f"data features: {features}")

    for row in data.itertuples():
        data_request = [x for x in row]
        request = {"data": [data_request[-30:]], "features": features}
        logger.info(f"request {request}")
        response = requests.get(url, json=request, )
        logger.info(f"CODE: {response.status_code}| RESPONSE: {response.json()}")


if __name__ == "__main__":
    main()