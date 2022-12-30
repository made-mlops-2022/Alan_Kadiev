import pytest
from airflow.models import DagBag


@pytest.fixture()
def degbag():
    return DagBag(dag_folder="/Users/user/Desktop/MLInProd/dz1/airflow_ml_dags", include_examples=False)


def test_dag_imports(degbag):
    assert degbag.dags is not None
    assert degbag.import_errors == {}


def test_download_dag(degbag):
    assert "download_dag" in degbag.dags
    assert len(degbag.dags["download_dag"].tasks) == 4


def test_train_dag(degbag):
    assert "train_dag" in degbag.dags
    assert len(degbag.dags["train_dag"].tasks) == 6


def test_predict_dag(degbag):
    assert "predict_dag" in degbag.dags
    assert len(degbag.dags["predict_dag"].tasks) == 3