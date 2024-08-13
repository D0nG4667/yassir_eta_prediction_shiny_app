import os
from dotenv import load_dotenv

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.coder import PickleCoder
from fastapi_cache.decorator import cache
import logging

from pydantic import BaseModel, Field
from typing import List, Union, Optional
from datetime import datetime

from sklearn.pipeline import Pipeline
import joblib

import pandas as pd

import httpx
from io import BytesIO


from utils.config import (
    ONE_DAY_SEC,
    ONE_WEEK_SEC,
    ENV_PATH,
    DESCRIPTION,
    ALL_MODELS
)

load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    FastAPICache.init(InMemoryBackend())
    yield


# FastAPI Object
app = FastAPI(
    title='Yassir Eta Prediction',
    version='1.0.0',
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get('/favicon.ico', include_in_schema=False)
@cache(expire=ONE_WEEK_SEC, namespace='eta_favicon')  # Cache for 1 week
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "assets", file_name)
    return FileResponse(path=file_path, headers={"Content-Disposition": "attachment; filename=" + file_name})


# API input features


class EtaFeatures(BaseModel):
    timestamp: List[datetime] = Field(
        description="Timestamp: Time that the trip was started")
    origin_lat: List[float] = Field(
        description="Origin_lat: Origin latitude (in degrees)")
    origin_lon: List[float] = Field(
        description="Origin_lon: Origin longitude (in degrees)")
    destination_lat: List[float] = Field(
        description="Destination_lat: Destination latitude (in degrees)")
    destination_lon: List[float] = Field(
        description="Destination_lon: Destination longitude (in degrees)")
    trip_distance: List[float] = Field(
        description="Trip_distance: Distance in meters on a driving route")


class Url(BaseModel):
    url: str
    pipeline_url: str


class ResultData(BaseModel):
    prediction: List[float]


class PredictionResponse(BaseModel):
    execution_msg: str
    execution_code: int
    result: ResultData


class ErrorResponse(BaseModel):
    execution_msg: str
    execution_code: int
    error: Optional[str]


logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Load the model pipelines and encoder
# Cache for 1 day
@cache(expire=ONE_DAY_SEC, namespace='pipeline_resource', coder=PickleCoder)
async def load_pipeline(pipeline_url: Url) -> Pipeline:
    async def url_to_data(url: Url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()  # Ensure we catch any HTTP errors
            # Convert response content to BytesIO object
            data = BytesIO(response.content)
            return data

    pipeline = None
    try:
        pipeline: Pipeline = joblib.load(await url_to_data(pipeline_url))
    except Exception as e:
        logging.error(
            "Omg, an error occurred in loading the pipeline resources: %s", e)
    finally:
        return pipeline


# Endpoints

# Status endpoint: check if api is online
@app.get('/')
@cache(expire=ONE_WEEK_SEC, namespace='eta_status_check')  # Cache for 1 week
async def status_check():
    return {"Status": "API is online..."}


@cache(expire=ONE_DAY_SEC, namespace='pipeline_regressor')  # Cache for 1 day
async def pipeline_regressor(pipeline: Pipeline, data: EtaFeatures) -> Union[ErrorResponse, PredictionResponse]:
    msg = 'Execution failed'
    code = 0
    output = ErrorResponse(**{'execution_msg': msg,
                              'execution_code': code, 'error': None})

    try:
        # Create dataframe
        df = pd.DataFrame.from_dict(data.__dict__)

        # Make prediction
        preds = pipeline.predict(df)
        predictions = [float(pred) for pred in preds]

        result = ResultData(**{"prediction": predictions})

        msg = 'Execution was successful'
        code = 1
        output = PredictionResponse(
            **{'execution_msg': msg,
               'execution_code': code, 'result': result}
        )

    except Exception as e:
        error = f"Omg, pipeline regressor failure. {e}"
        output = ErrorResponse(**{'execution_msg': msg,
                                  'execution_code': code, 'error': error})

    finally:
        return output


@app.post('/api/v1/eta/prediction', tags=['All Models'])
async def query_eta_prediction(data: EtaFeatures, model: str = Query('RandomForestRegressor', enum=list(ALL_MODELS.keys()))) -> Union[ErrorResponse, PredictionResponse]:
    pipeline_url: Url = ALL_MODELS[model]
    pipeline = await load_pipeline(pipeline_url)
    output = await pipeline_regressor(pipeline, data)
    return output
