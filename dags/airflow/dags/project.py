import mlflow
import os
import logging
import pandas as pd
import pytz
import json
import io

from airflow.models import DAG, Variable
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Union

from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from mlflow.models import infer_signature

BUCKET = Variable.get("S3_BUCKET")
TIMEZONE = pytz.timezone('Europe/Moscow')
DEFAULT_ARGS = {
    "owner": "Alex Trigolos",
    "email": "alexvlg34@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="Alex_Trigolos",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)

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

for key in [
    "MLFLOW_TRACKING_URI",
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
]:
    os.environ[key] = Variable.get(key)

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


def s3_save_file(file_name: str, data: Union[pd.DataFrame, pd.Series]) -> None:
    s3_hook = S3Hook("s3_connection")
    buffer = io.BytesIO()
    data.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f"AlexTrigolos/project/datasets/{file_name}.pkl", bucket_name=BUCKET, replace=True)


def s3_download_file(file_name: str) -> Union[pd.DataFrame, pd.Series]:
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"AlexTrigolos/project/datasets/{file_name}.pkl", bucket_name=BUCKET)
    _LOG.info(file)
    data = pd.read_pickle(file)
    return data


def init(**kwargs) -> None:
    _LOG.info("Train pipeline started.")
    _LOG.info(mlflow.get_registry_uri())
    start_time = datetime.now(TIMEZONE)

    exp_id = mlflow.create_experiment(name="Alex Trigolos project")
    with mlflow.start_run(run_name="ursatap", experiment_id=exp_id, description="parent") as parent_run:
        parent_run_id = parent_run.info.run_id

    kwargs['ti'].xcom_push(key='start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='exp_id', value=exp_id)
    kwargs['ti'].xcom_push(key='parent_run_id', value=parent_run_id)


def get_data(**kwargs) -> None:
    _LOG.info("Get data...")

    start_time = datetime.now(TIMEZONE)
    data = pd.read_csv('https://raw.githubusercontent.com/edaehn/python_tutorials/main/titanic/train.csv', index_col=0)
    end_time = datetime.now(TIMEZONE)
    
    kwargs['ti'].xcom_push(key='get_start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='get_end_time', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='dataset_size', value=data.shape)

    s3_save_file('dataset', data)

    _LOG.info("Finish get data.")


def prepare_data(**kwargs) -> None:
    _LOG.info("Prepare data...")

    data = s3_download_file('dataset')

    start_time = datetime.now(TIMEZONE)

    prepare_data = data.fillna({"Embarked" : "S"})
    prepare_data['Age'] = prepare_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    prepare_data = pd.get_dummies(prepare_data, columns=["Sex", "Pclass", "Embarked"])
    prepared_data = prepare_data.drop(["Name", "Ticket", "Cabin"], axis=1)
    X = prepared_data.drop("Survived", axis=1)
    y = prepared_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    end_time = datetime.now(TIMEZONE)

    kwargs['ti'].xcom_push(key='prepare_start_time', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='prepare_end_time', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    kwargs['ti'].xcom_push(key='features', value=X_train.columns.tolist())
    kwargs['ti'].xcom_push(key='prepared_size', value=prepared_data.shape)

    s3_save_file('X_train', X_train)
    s3_save_file('X_val', X_val)
    s3_save_file('X_test', X_test)
    s3_save_file('y_train', y_train)
    s3_save_file('y_val', y_val)
    s3_save_file('y_test', y_test)

    _LOG.info("Prepared data.")


def train_model(**kwargs) -> None:
    _LOG.info("Train model...")

    exp_id = kwargs['ti'].xcom_pull(key='exp_id')
    parent_run_id = kwargs['ti'].xcom_pull(key='parent_run_id')

    X_train = s3_download_file('X_train')
    X_val = s3_download_file('X_val')
    X_test = s3_download_file('X_test')
    y_train = s3_download_file('y_train')
    y_val = s3_download_file('y_val')
    y_test = s3_download_file('y_test')

    model_name = kwargs["model_name"]
    model = MODELS[model_name]["model"]
    param_grid = MODELS[model_name]["params"]

    start_time = datetime.now(TIMEZONE)

    with mlflow.start_run(parent_run_id=parent_run_id, run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
        grid_search.fit(X_train, y_train)

        end_time = datetime.now(TIMEZONE)
    
        best_model = grid_search.best_estimator_
        prediction = best_model.predict(X_val)

        eval_df = X_val.copy()
        eval_df["target"] = y_val

        signature = infer_signature(X_test, prediction)
        model_info = mlflow.sklearn.log_model(best_model, model_name, signature=signature, registered_model_name=f"sk-learn-{model_name}-model")
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="classifier",
            evaluators=["default"],
        )

        kwargs['ti'].xcom_push(key=f'training_start_time_{model_name}', value=start_time.strftime("%Y-%m-%d %H:%M:%S"))
        kwargs['ti'].xcom_push(key=f'training_end_time_{model_name}', value=end_time.strftime("%Y-%m-%d %H:%M:%S"))
        kwargs['ti'].xcom_push(key=f'best_params_{model_name}', value=grid_search.best_params_)

    _LOG.info("Finish model train.")


def save_results(**kwargs) -> None:
    _LOG.info("Save metrics...")

    s3_hook = S3Hook("s3_connection")

    dtc, rfc, gbc = "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"
    metrics = {
        'start_time': kwargs['ti'].xcom_pull(key='start_time'),
        'get_start_time': kwargs['ti'].xcom_pull(key='get_start_time'),
        'get_end_time': kwargs['ti'].xcom_pull(key='get_end_time'),
        'dataset_size': kwargs['ti'].xcom_pull(key='dataset_size'),
        'prepare_start_time': kwargs['ti'].xcom_pull(key='prepare_start_time'),
        'prepare_end_time': kwargs['ti'].xcom_pull(key='prepare_end_time'),
        'prepared_size': kwargs['ti'].xcom_pull(key='prepared_size'),
        'features': kwargs['ti'].xcom_pull(key='features'),
        f'training_start_time_{dtc}': kwargs['ti'].xcom_pull(key=f'training_start_time_{dtc}'),
        f'training_end_time_{dtc}': kwargs['ti'].xcom_pull(key=f'training_end_time_{dtc}'),
        f'training_start_time_{rfc}': kwargs['ti'].xcom_pull(key=f'training_start_time_{rfc}'),
        f'training_end_time_{rfc}': kwargs['ti'].xcom_pull(key=f'training_end_time_{rfc}'),
        f'training_start_time_{gbc}': kwargs['ti'].xcom_pull(key=f'training_start_time_{gbc}'),
        f'training_end_time_{gbc}': kwargs['ti'].xcom_pull(key=f'training_end_time_{gbc}'),
        f'best_params_{dtc}': kwargs['ti'].xcom_pull(key=f'best_params_{dtc}'),
        f'best_params_{rfc}': kwargs['ti'].xcom_pull(key=f'best_params_{rfc}'),
        f'best_params_{gbc}': kwargs['ti'].xcom_pull(key=f'best_params_{gbc}')
    }

    buffer = io.BytesIO()
    buffer.write(json.dumps(metrics).encode())
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f"AlexTrigolos/project/results/metrics.pkl", bucket_name=BUCKET, replace=True)

    _LOG.info("DONE!!!")


task_init = PythonOperator(task_id="task_init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id='task_get_data', python_callable=get_data, dag=dag)

task_prepare_data = PythonOperator(task_id='task_prepare_data', python_callable=prepare_data, dag=dag)

training_model_tasks = [
    PythonOperator(task_id='training_dtc_model_tasks', python_callable=train_model, dag=dag, op_kwargs={ "model_name": "DecisionTreeClassifier" }),
    PythonOperator(task_id='training_rfc_model_tasks', python_callable=train_model, dag=dag, op_kwargs={ "model_name": "RandomForestClassifier" }),
    PythonOperator(task_id='training_gbc_model_tasks', python_callable=train_model, dag=dag, op_kwargs={ "model_name": "GradientBoostingClassifier" })
]

task_save_results = PythonOperator(task_id='task_save_results', python_callable=save_results, dag=dag)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results