"""Stateless HTTP adapter for BasinCast Engine v1.1 automatic horizon.

This module only validates requests, runs the tested Python engine and exports its
outputs. No AI, weather service, database or probabilistic model is required.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import tempfile
import zipfile
from threading import BoundedSemaphore
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from basincast import __version__ as engine_version
from basincast.cli import run_forecast
from basincast.engine import ANALYSIS_CEILING_MONTHS

LOGGER = logging.getLogger("basincast_api")
MAX_UPLOAD_BYTES = int(os.getenv("BASINCAST_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_CONCURRENT_JOBS = max(1, min(4, int(os.getenv("BASINCAST_MAX_CONCURRENT_JOBS", "1"))))
MAX_SOURCE_ROWS = int(os.getenv("BASINCAST_MAX_SOURCE_ROWS", "6000"))
SLOTS = BoundedSemaphore(MAX_CONCURRENT_JOBS)
API_KEY = os.getenv("BASINCAST_API_KEY", "")
REQUIRE_API_KEY = os.getenv("BASINCAST_REQUIRE_API_KEY", "0") == "1"

app = FastAPI(
    title="BasinCast Forecast API",
    version="1.1.0",
    description=(
        "Stateless, reproducible monthly forecasts for user-provided observations. "
        "No user-selected forecast horizon: empirical reliability is determined "
        "from rolling-origin KGE validation and a continuous-month rule. "
        "Exploratory continuation is distinct from a validated prediction."
    ),
)

allowed_origins = [
    value.strip() for value in os.getenv("BASINCAST_CORS_ORIGINS", "").split(",")
    if value.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

AssetType = Literal[
    "river", "aquifer", "reservoir", "agricultural_demand", "urban_demand", "custom"
]
DataStatus = Literal[
    "OBSERVED_REPORTED", "USER_PROVIDED_UNVERIFIED", "RECONSTRUCTED",
    "ESTIMATED", "SYNTHETIC_DEMO",
]
Aggregation = Literal["mean", "sum", "last"]


def verify_api_key(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
    """A shared key is stored server-side and is never placed in browser code."""
    if REQUIRE_API_KEY and not API_KEY:
        raise HTTPException(status_code=503, detail="Server API key is not configured")
    if API_KEY and (not x_api_key or not hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "api_version": "1.1.0", "engine_version": engine_version}


@app.get("/v1/info")
def info() -> dict:
    return {
        "engine_version": engine_version,
        "asset_types": ["river", "aquifer", "reservoir", "agricultural_demand", "urban_demand", "custom"],
        "formats": ["csv", "xlsx"],
        "required_columns": ["date", "value"],
        "horizon_selection": "AUTOMATIC_KGE_CONTIGUOUS",
        "analysis_ceiling_months": ANALYSIS_CEILING_MONTHS,
        "kge_threshold": 0.60,
        "forecast_phases": ["PREDICTION", "PLANNING_GUIDANCE"],
        "external_meteorology_required": False,
        "climate_scenarios_generated_by_this_api": False,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_source_rows": MAX_SOURCE_ROWS,
        "privacy": "Uploads and results exist only during the request; no server-side storage",
    }


def _records(path: Path) -> list[dict]:
    # pandas outputs date strings and JSON null for CSV missing values.
    dataframe = pd.read_csv(path)
    return json.loads(dataframe.to_json(orient="records", date_format="iso"))


def _check_input_size_and_rows(uploaded_path: Path) -> None:
    """Limit CPU and memory use on the free host before parsing a workbook."""
    if uploaded_path.suffix == ".csv":
        with uploaded_path.open("rb") as source:
            if sum(1 for _ in source) > MAX_SOURCE_ROWS + 1:
                raise HTTPException(status_code=413, detail="Too many source CSV rows")
        return
    # XLSX is a ZIP: limit inflated payloads before openpyxl sees them.
    try:
        with zipfile.ZipFile(uploaded_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 40 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="XLSX decompressed content exceeds the free-tier limit")
        from openpyxl import load_workbook
        workbook = load_workbook(uploaded_path, read_only=True, data_only=True)
        try:
            sheet = workbook.worksheets[0]
            for count, _ in enumerate(sheet.iter_rows(values_only=True), start=1):
                if count > MAX_SOURCE_ROWS + 1:
                    raise HTTPException(status_code=413, detail="Too many source XLSX rows")
        finally:
            workbook.close()
    except (zipfile.BadZipFile, OSError) as exc:
        raise HTTPException(status_code=422, detail="Invalid XLSX file") from exc


def _execute_forecast(
    file: UploadFile,
    series_id: str,
    asset_type: AssetType,
    unit: str,
    date_column: str,
    value_column: str,
    data_status: DataStatus,
    source_description: str,
    aggregation: Aggregation | None,
    decimal_comma: bool,
    capacity: float | None,
    as_zip: bool,
) -> dict | bytes:
    if not series_id or len(series_id) > 101:
        raise HTTPException(status_code=422, detail="Invalid series identifier")
    if not unit.strip() or len(unit) > 64:
        raise HTTPException(status_code=422, detail="The physical unit is required")
    if not date_column or not value_column or len(date_column) > 100 or len(value_column) > 100:
        raise HTTPException(status_code=422, detail="Invalid CSV column name")
    if not source_description.strip() or len(source_description) > 300:
        raise HTTPException(status_code=422, detail="Source description is required (max 300 characters)")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=415, detail="Only monthly CSV and XLSX files are accepted")
    if not SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server busy; try again after the current prediction finishes")
    try:
        with tempfile.TemporaryDirectory(prefix="basincast_api_") as working:
            root = Path(working)
            uploaded_path = root / ("upload" + suffix)
            total = 0
            with uploaded_path.open("wb") as destination:
                while True:
                    block = file.file.read(256 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Maximum upload size is {MAX_UPLOAD_BYTES} bytes",
                        )
                    destination.write(block)
            if total == 0:
                raise HTTPException(status_code=422, detail="Empty uploaded file")
            _check_input_size_and_rows(uploaded_path)
            try:
                result = run_forecast(
                    uploaded_path,
                    root / "result",
                    series_id=series_id,
                    asset_type=asset_type,
                    unit=unit,
                    date_column=date_column,
                    value_column=value_column,
                    source_description=source_description,
                    data_status=data_status,
                    aggregation=aggregation,
                    decimal_comma=decimal_comma,
                    capacity=capacity,
                )
            except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
                message = str(exc).replace(str(root), "<temporary directory>")
                raise HTTPException(status_code=422, detail=message[:700]) from exc
            if as_zip:
                return Path(result["result_zip"]).read_bytes()
            summary = json.loads((root / "result" / "summary.json").read_text(encoding="utf-8"))
            # The output ZIP path exists only while the request is being processed;
            # do not return temporary local paths to web clients.
            return {
                "summary": summary,
                "forecast": _records(root / "result" / "forecast_all_candidates.csv"),
                "supported_forecast": _records(root / "result" / "supported_forecast.csv"),
                "planning_guidance": _records(root / "result" / "planning_guidance.csv"),
                "metrics": _records(root / "result" / "metrics_by_horizon.csv"),
                "zip_sha256": hashlib.sha256(Path(result["result_zip"]).read_bytes()).hexdigest(),
                "interpretation": (
                    "The model determines the continuous supported prediction horizon "
                    "from KGE and sample support. All later numerical values are exploratory "
                    "planning guidance, never validated predictions or management directives. "
                    "Compare RMSE against persistence and seasonal baselines separately."
                ),
            }
    finally:
        SLOTS.release()


# FastAPI exposes these parameters as form fields alongside the file upload.
# The two routes use the same engine and share the same validation rules.
def _form_forecast(
    file: UploadFile,
    series_id: str,
    asset_type: AssetType,
    unit: str,
    date_column: str,
    value_column: str,
    data_status: DataStatus,
    source_description: str,
    aggregation: Aggregation | None,
    decimal_comma: bool,
    capacity: float | None,
    as_zip: bool,
) -> dict | bytes:
    return _execute_forecast(
        file=file,
        series_id=series_id,
        asset_type=asset_type,
        unit=unit,
        date_column=date_column,
        value_column=value_column,
        data_status=data_status,
        source_description=source_description,
        aggregation=aggregation,
        decimal_comma=decimal_comma,
        capacity=capacity,
        as_zip=as_zip,
    )


@app.post("/v1/forecast", dependencies=[Depends(verify_api_key)])
def forecast(
    file: Annotated[UploadFile, File(description="One monthly CSV or XLSX containing date and value")],
    series_id: Annotated[str, Form()] = "Site_001",
    asset_type: Annotated[AssetType, Form()] = "river",
    unit: Annotated[str, Form()] = "m3/s",
    date_column: Annotated[str, Form()] = "date",
    value_column: Annotated[str, Form()] = "value",
    data_status: Annotated[DataStatus, Form()] = "USER_PROVIDED_UNVERIFIED",
    source_description: Annotated[str, Form()] = "USER_UPLOAD_UNVERIFIED",
    aggregation: Annotated[Aggregation | None, Form()] = None,
    decimal_comma: Annotated[bool, Form()] = False,
    capacity: Annotated[float | None, Form()] = None,
) -> dict:
    return _form_forecast(
        file, series_id, asset_type, unit, date_column,
        value_column, data_status, source_description, aggregation,
        decimal_comma, capacity, False,
    )


@app.post("/v1/forecast.zip", dependencies=[Depends(verify_api_key)])
def forecast_zip(
    file: Annotated[UploadFile, File(description="One monthly CSV or XLSX containing date and value")],
    series_id: Annotated[str, Form()] = "Site_001",
    asset_type: Annotated[AssetType, Form()] = "river",
    unit: Annotated[str, Form()] = "m3/s",
    date_column: Annotated[str, Form()] = "date",
    value_column: Annotated[str, Form()] = "value",
    data_status: Annotated[DataStatus, Form()] = "USER_PROVIDED_UNVERIFIED",
    source_description: Annotated[str, Form()] = "USER_UPLOAD_UNVERIFIED",
    aggregation: Annotated[Aggregation | None, Form()] = None,
    decimal_comma: Annotated[bool, Form()] = False,
    capacity: Annotated[float | None, Form()] = None,
) -> Response:
    payload = _form_forecast(
        file, series_id, asset_type, unit, date_column,
        value_column, data_status, source_description, aggregation,
        decimal_comma, capacity, True,
    )
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="basincast_forecast.zip"'},
    )
