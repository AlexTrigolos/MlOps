import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = 'testmlops'
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Alex Trigolos",
    "email": "alexvlg34@gmail.com",
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="train",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)


def init() -> None:
    _LOG.info("Train pipeline started.")



def get_data_from_sklearn() -> None:
    # Получим датасет California housing
    housing = fetch_california_housing(as_frame=True)
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохранить файл в формате pkl на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="2024/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    _LOG.info("Data downloaded.")


def prepare_data() -> None:

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key="2024/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Сохранить готовые данные на S3
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"2024/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    _LOG.info("Data prepared.")


def train_model() -> None:

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"2024/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)

    # Обучить модель
    model = RandomForestRegressor()
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    # Посчитать метрики
    result = {}
    result["r2_score"] = r2_score(data["y_test"], prediction)
    result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
    result["mae"] = median_absolute_error(data["y_test"], prediction)

    # Сохранить результат на S3
    date = datetime.now().strftime("%Y_%m_%d_%H")
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(result).encode())
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"2024/metrics/{date}.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info("Model trained.")


def save_results() -> None:
    print("Success")
    _LOG.info("Success.")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_download_data = PythonOperator(
    task_id="download_data", python_callable=get_data_from_sklearn, dag=dag
)

task_prepare_data = PythonOperator(
    task_id="data_preparation", python_callable=prepare_data, dag=dag
)

task_train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag
)

task_save_results = PythonOperator(
    task_id="save_results", python_callable=train_model, dag=dag
)

task_init >> task_download_data >> task_prepare_data >> task_train_model >> task_save_results
