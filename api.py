import tensorflow as tf
from enum import Enum
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from pathlib import Path
import shutil

from model.result import Result
from extract.feature_extraction_classification import ExtractorClassification

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#TODO add commentary

app = FastAPI()

#extractor = ExtractorClassification(r'../etc/config.yaml')
extractor = ExtractorClassification(r'C:/Users/Mimnat/Documents/DAS/etc/config.yaml')

class ModelName(str, Enum):
    classification = "classification"
    AE_convolutif = "AE_convolutif"


class ModelInput(BaseModel):
    path: UploadFile = File(...)
    machine: str
    id_m: str


@app.get("/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.classification:
        return {"model_name": model_name, "message": "Classification FTW!"}
    if model_name == ModelName.AE_convolutif:
        return {"model_name": model_name, "message": "Autoencoder FTW!"}

    return {"model_name": model_name, "message": "Have some residuals"}


@app.post("/{model_name}/result")
def get_result(machine: str, id_m: str, path: UploadFile = File(...)):
    result = Result(machine, id_m, extractor)
    try:
        suffix = Path(path.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(path.file, tmp)
    finally:
        path.file.close()
    out = result.predict(tmp.name)

    if out == 0:
        return f"Audio '{path.filename}' sounds normal for machine {machine} {id_m}"
    else:
        return f"Audio '{path.filename}' sounds anormal for machine {machine} {id_m}"

