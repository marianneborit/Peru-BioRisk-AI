"""
Peru BioRisk AI — Master Airflow DAG
Runs weekly: ingest all data domains → ETL → feature engineering → model inference → alerts.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "biorisk-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=4),
}

with DAG(
    dag_id="peru_biorisk_weekly_pipeline",
    default_args=default_args,
    description="Weekly biological risk update pipeline",
    schedule_interval="0 6 * * 1",  # every Monday at 06:00 UTC
    start_date=days_ago(7),
    catchup=False,
    max_active_runs=1,
    tags=["biorisk", "production"],
) as dag:

    # ── 1. Data ingestion ─────────────────────────────────────────────────────
    ingest_senamhi = PythonOperator(
        task_id="ingest_senamhi",
        python_callable=lambda **ctx: __import__(
            "src.ingestion.senamhi", fromlist=["extract"]
        ).extract(
            start_date=ctx["data_interval_start"].date(),
            end_date=ctx["data_interval_end"].date(),
            output_dir="/opt/airflow/data/raw/climatic",
        ),
    )

    ingest_era5 = PythonOperator(
        task_id="ingest_era5",
        python_callable=lambda **ctx: __import__(
            "src.ingestion.era5", fromlist=["extract"]
        ).extract(
            start_date=ctx["data_interval_start"].date(),
            end_date=ctx["data_interval_end"].date(),
            output_dir="/opt/airflow/data/raw/climatic",
        ),
    )

    ingest_modis = PythonOperator(
        task_id="ingest_modis",
        python_callable=lambda **ctx: __import__(
            "src.ingestion.modis", fromlist=["extract"]
        ).extract(
            start_date=ctx["data_interval_start"].date(),
            end_date=ctx["data_interval_end"].date(),
            output_dir="/opt/airflow/data/raw/satellite",
        ),
    )

    ingest_cdc = PythonOperator(
        task_id="ingest_cdc_peru",
        python_callable=lambda **ctx: __import__(
            "src.ingestion.cdc_peru", fromlist=["extract"]
        ).extract(
            start_date=ctx["data_interval_start"].date(),
            end_date=ctx["data_interval_end"].date(),
            output_dir="/opt/airflow/data/raw/epidemiological",
        ),
    )

    ingest_geobosques = PythonOperator(
        task_id="ingest_geobosques",
        python_callable=lambda **ctx: __import__(
            "src.ingestion.geobosques", fromlist=["extract"]
        ).extract(
            start_date=ctx["data_interval_start"].date(),
            end_date=ctx["data_interval_end"].date(),
            output_dir="/opt/airflow/data/raw/landuse",
        ),
    )

    # ── 2. ETL: Reproject + align ─────────────────────────────────────────────
    etl_reproject = BashOperator(
        task_id="etl_reproject",
        bash_command=(
            "python /opt/airflow/scripts/run_etl.py "
            "--step reproject "
            "--start {{ data_interval_start.date() }} "
            "--end {{ data_interval_end.date() }}"
        ),
    )

    # ── 3. Feature engineering ────────────────────────────────────────────────
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            "python /opt/airflow/scripts/run_features.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }}"
        ),
    )

    # ── 4. Data validation ────────────────────────────────────────────────────
    validate_features = BashOperator(
        task_id="validate_features",
        bash_command=(
            "python /opt/airflow/scripts/run_validation.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }}"
        ),
    )

    # ── 5. Model inference ────────────────────────────────────────────────────
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            "python /opt/airflow/scripts/run_inference.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }} "
            "--horizons 4,8,12"
        ),
    )

    # ── 6. Update risk map in PostGIS ─────────────────────────────────────────
    update_risk_map = BashOperator(
        task_id="update_risk_map_postgis",
        bash_command=(
            "python /opt/airflow/scripts/update_risk_map.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }}"
        ),
    )

    # ── 7. Alert generation ───────────────────────────────────────────────────
    generate_alerts = BashOperator(
        task_id="generate_alerts",
        bash_command=(
            "python /opt/airflow/scripts/generate_alerts.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }}"
        ),
    )

    # ── 8. Notify stakeholders ────────────────────────────────────────────────
    notify = BashOperator(
        task_id="notify_stakeholders",
        bash_command=(
            "python /opt/airflow/scripts/notify.py "
            "--week {{ data_interval_start.strftime('%G-W%V') }}"
        ),
    )

    # ── DAG dependencies ──────────────────────────────────────────────────────
    [ingest_senamhi, ingest_era5, ingest_modis] >> etl_reproject
    [ingest_cdc, ingest_geobosques] >> etl_reproject
    etl_reproject >> feature_engineering >> validate_features
    validate_features >> model_inference >> update_risk_map >> generate_alerts >> notify
