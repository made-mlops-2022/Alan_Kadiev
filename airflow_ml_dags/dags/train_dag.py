from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from pendulum import today
from datetime import timedelta
from airflow.models import Variable
from docker.types import Mount

MOUNT_DIR="/Users/user/Desktop/MLInProd/dz1/airflow_ml_dags"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        dag_id="train_dag",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=today("UTC").add(days=-5),
) as dag:

    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath="data/{{ ds }}/generate/data.csv"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        task_id="docker-airflow-preprocess",
        command=["/data/{{ ds }}/generate/", "/data/{{ ds }}/processed/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data",
                      type='bind')]
    )

    split = DockerOperator(
        image="airflow-split",
        task_id="docker-airflow-split",
        command=["/data/{{ ds }}/processed/", "/data/{{ ds }}/split/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data",
                      type='bind')]
    )

    train = DockerOperator(
        image="airflow-model-train",
        task_id="docker-airflow-train",
        command=["/data/{{ ds }}/split/", "/data/{{ ds }}/models/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data",
                      type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        task_id="docker-airflow-validate",
        command=["/data/{{ ds }}/split/", "/data/{{ ds }}/models/", "/data/{{ ds }}/metrics/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data",
                      type='bind')]
    )

    finish = EmptyOperator(task_id="finish_train_model")

    data_sensor >> preprocess >> split >> train >> validate >> finish