import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import pytz
import re

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import NoReturn, Union
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

BUCKET = Variable.get("S3_BUCKET")
TIMEZONE = pytz.timezone('Europe/Moscow')
MODELS = {
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": range(50, 500, 50),
            "max_depth": [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30],
            "min_samples_split": [1, 2, 3, 5, 7, 8, 10]
        }
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30],
            "min_samples_split": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "params": {
            "n_estimators": range(50, 500, 50),
            "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1],
            "max_depth": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        }
    }
}

DEFAULT_ARGS = {
    "owner": "Alex Trigolos",
    "email": "alexvlg34@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


def s3_save_file(model_name: str, file_name: str, data: Union[pd.DataFrame, pd.Series]) -> None:
    s3_hook = S3Hook("s3_connection")
    buffer = io.BytesIO()
    data.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f"AlexTrigolos/{model_name}/datasets/{file_name}.pkl", bucket_name=BUCKET, replace=True)


def s3_download_file(model_name: str, file_name: str) -> Union[pd.DataFrame, pd.Series]:
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"AlexTrigolos/{model_name}/datasets/{file_name}.pkl", bucket_name=BUCKET)
    _LOG.info(file)
    data = pd.read_pickle(file)
    return data


def init(**kwargs) -> None:
    _LOG.info("Train pipeline started.")
    start_time = datetime.now(TIMEZONE)
    kwargs['ti'].xcom_push(key='model_name', value=kwargs['model_name'])
    kwargs['ti'].xcom_push(key='start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))


def get_data(**kwargs) -> None:
    _LOG.info("Get data...")

    start_time = datetime.now(TIMEZONE)
    data = pd.read_csv('https://raw.githubusercontent.com/edaehn/python_tutorials/main/titanic/train.csv', index_col=0)
    end_time = datetime.now(TIMEZONE)
    
    kwargs['ti'].xcom_push(key='get_start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='get_end_time', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='dataset_size', value=data.shape)

    s3_save_file(kwargs['model_name'], 'dataset', data)

    _LOG.info("Finish get data.")


def prepare_data(**kwargs) -> None:
    _LOG.info("Prepare data...")

    data = s3_download_file(kwargs['model_name'], 'dataset')

    start_time = datetime.now(TIMEZONE)

    prepare_data = data.fillna({"Embarked" : "S"})
    prepare_data['Age'] = prepare_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    prepare_data = pd.get_dummies(prepare_data, columns=["Sex", "Pclass", "Embarked"])
    prepared_data = prepare_data.drop(["Name", "Ticket", "Cabin"], axis=1)
    X = prepared_data.drop("Survived", axis=1)
    y = prepared_data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    end_time = datetime.now(TIMEZONE)

    kwargs['ti'].xcom_push(key='prepare_start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='prepare_end_time', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='features', value=X_train.columns.tolist())
    kwargs['ti'].xcom_push(key='prepared_size', value=prepared_data.shape)

    s3_save_file(kwargs['model_name'], 'X_train', X_train)
    s3_save_file(kwargs['model_name'], 'X_test', X_test)
    s3_save_file(kwargs['model_name'], 'y_train', y_train)
    s3_save_file(kwargs['model_name'], 'y_test', y_test)

    _LOG.info("Prepared data.")


def train_model(**kwargs) -> None:
    _LOG.info("Train model...")

    X_train = s3_download_file(kwargs['model_name'], 'X_train')
    X_test = s3_download_file(kwargs['model_name'], 'X_test')
    y_train = s3_download_file(kwargs['model_name'], 'y_train')
    y_test = s3_download_file(kwargs['model_name'], 'y_test')

    model = MODELS[kwargs["model_name"]]["model"]
    param_grid = MODELS[kwargs["model_name"]]["params"]

    start_time = datetime.now(TIMEZONE)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2)
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now(TIMEZONE)

    y_pred = grid_search.best_estimator_.predict(X_test)

    kwargs['ti'].xcom_push(key='training_start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='training_end_time', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='accuracy_score', value=accuracy_score(y_test, y_pred))
    kwargs['ti'].xcom_push(key='r2_score', value=r2_score(y_test, y_pred))
    kwargs['ti'].xcom_push(key='mean_squared_error', value=mean_squared_error(y_test, y_pred))
    kwargs['ti'].xcom_push(key='best_params', value=grid_search.best_params_)

    _LOG.info("Finish model train.")


def save_results(**kwargs) -> None:
    _LOG.info("Save metrics...")

    s3_hook = S3Hook("s3_connection")

    metrics = {
        'model_name': kwargs['ti'].xcom_pull(key='model_name'),
        'start_time': kwargs['ti'].xcom_pull(key='start_time'),
        'get_start_time': kwargs['ti'].xcom_pull(key='get_start_time'),
        'get_end_time': kwargs['ti'].xcom_pull(key='get_end_time'),
        'dataset_size': kwargs['ti'].xcom_pull(key='dataset_size'),
        'prepare_start_time': kwargs['ti'].xcom_pull(key='prepare_start_time'),
        'prepare_end_time': kwargs['ti'].xcom_pull(key='prepare_end_time'),
        'prepared_size': kwargs['ti'].xcom_pull(key='prepared_size'),
        'features': kwargs['ti'].xcom_pull(key='features'),
        'training_start_time': kwargs['ti'].xcom_pull(key='training_start_time'),
        'training_end_time': kwargs['ti'].xcom_pull(key='training_end_time'),
        'accuracy_score': kwargs['ti'].xcom_pull(key='accuracy_score'),
        'r2_score': kwargs['ti'].xcom_pull(key='r2_score'),
        'mean_squared_error': kwargs['ti'].xcom_pull(key='mean_squared_error'),
        'best_params': kwargs['ti'].xcom_pull(key='best_params')
    }

    buffer = io.BytesIO()
    buffer.write(json.dumps(metrics).encode())
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f"AlexTrigolos/{kwargs['model_name']}/results/metrics.pkl", bucket_name=BUCKET, replace=True)

    _LOG.info("DONE!!!")


decision_tree_classifier_dag = DAG(
    dag_id="decision_tree_classifier_dag",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)

op_kwargs = {"model_name": "DecisionTreeClassifier"}

dtc_init = PythonOperator(task_id="dtc_init", python_callable=init, dag=decision_tree_classifier_dag, op_kwargs=op_kwargs)

dtc_get_data = PythonOperator(task_id='dtc_get_data', python_callable=get_data, dag=decision_tree_classifier_dag, op_kwargs=op_kwargs)

dtc_prepare_data = PythonOperator(task_id='dtc_prepare_data', python_callable=prepare_data, dag=decision_tree_classifier_dag, op_kwargs=op_kwargs)

dtc_train_model = PythonOperator(task_id='dtc_train_model', python_callable=train_model, dag=decision_tree_classifier_dag, op_kwargs=op_kwargs)

dtc_save_results = PythonOperator(task_id='dtc_save_results', python_callable=save_results, dag=decision_tree_classifier_dag, op_kwargs=op_kwargs)

dtc_init >> dtc_get_data >> dtc_prepare_data >> dtc_train_model >> dtc_save_results


random_forest_classifier_dag = DAG(
    dag_id="random_forest_classifier_dag",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)

op_kwargs = {"model_name": "RandomForestClassifier"}

rfc_init = PythonOperator(task_id="rfc_init", python_callable=init, dag=random_forest_classifier_dag, op_kwargs=op_kwargs)

rfc_get_data = PythonOperator(task_id='rfc_get_data', python_callable=get_data, dag=random_forest_classifier_dag, op_kwargs=op_kwargs)

rfc_prepare_data = PythonOperator(task_id='rfc_prepare_data', python_callable=prepare_data, dag=random_forest_classifier_dag, op_kwargs=op_kwargs)

rfc_train_model = PythonOperator(task_id='rfc_train_model', python_callable=train_model, dag=random_forest_classifier_dag, op_kwargs=op_kwargs)

rfc_save_results = PythonOperator(task_id='rfc_save_results', python_callable=save_results, dag=random_forest_classifier_dag, op_kwargs=op_kwargs)

rfc_init >> rfc_get_data >> rfc_prepare_data >> rfc_train_model >> rfc_save_results


gradient_boosting_classifier_dag = DAG(
    dag_id="gradient_boosting_classifier_dag",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)

op_kwargs = {"model_name": "GradientBoostingClassifier"}

gbc_init = PythonOperator(task_id="gbc_init", python_callable=init, dag=gradient_boosting_classifier_dag, op_kwargs=op_kwargs)

gbc_get_data = PythonOperator(task_id='gbc_get_data', python_callable=get_data, dag=gradient_boosting_classifier_dag, op_kwargs=op_kwargs)

gbc_prepare_data = PythonOperator(task_id='gbc_prepare_data', python_callable=prepare_data, dag=gradient_boosting_classifier_dag, op_kwargs=op_kwargs)

gbc_train_model = PythonOperator(task_id='gbc_train_model', python_callable=train_model, dag=gradient_boosting_classifier_dag, op_kwargs=op_kwargs)

gbc_save_results = PythonOperator(task_id='gbc_save_results', python_callable=save_results, dag=gradient_boosting_classifier_dag, op_kwargs=op_kwargs)

gbc_init >> gbc_get_data >> gbc_prepare_data >> gbc_train_model >> gbc_save_results
