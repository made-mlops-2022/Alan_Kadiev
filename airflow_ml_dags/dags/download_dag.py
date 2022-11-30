from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator

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
        dag_id="download_dag",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=today("UTC").add(days=-7),
) as dag:

    start = EmptyOperator(task_id="start_generate_data")

    download = DockerOperator(
        image="airflow-download",
        task_id="docker-airflow-download",
        command=["/data/{{ ds }}/raw/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    generate = DockerOperator(
        image="airflow-generate",
        task_id="docker-airflow-generate",
        command=["/data/{{ ds }}/raw/", "/data/{{ ds }}/generate/"],
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    finish = EmptyOperator(task_id="finish_generate_data")

    start >> download >> generate >> finish