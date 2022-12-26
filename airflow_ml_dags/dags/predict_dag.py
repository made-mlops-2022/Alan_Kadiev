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
        dag_id="predict_dag",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=today("UTC").add(days=-3),
) as dag:

    model_sensor = FileSensor(
        task_id="data_sensor",
        filepath="data/{{ ds }}/models/model.pkl"
    )

    predict = DockerOperator(
        image="airflow-predict",
        task_id="docker-airflow-predict",
        command=["/data/{{ ds }}/raw/", "/data/{{ ds }}/processed/", "/data/{{ ds }}/models/",
                 "/data/{{ ds }}/prediction/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data",
                      type='bind')]
    )

    finish = EmptyOperator(task_id="finish_predict")

    model_sensor >> predict >> finish