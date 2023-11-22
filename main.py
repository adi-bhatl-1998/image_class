from typing import Union ,Annotated
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, File, UploadFile ,Request , BackgroundTasks ,HTTPException, Form
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,validators
from fastapi.staticfiles import StaticFiles
from typing import Optional
from typing import Union 
import asyncio
import cv2
import numpy as np
import albumentations as A
import torch 
import matplotlib.pyplot as plt 
from imp_funct import deploy_and_test 


app=FastAPI()
templates=Jinja2Templates(directory="Templates")
@app.get("/",response_class="HTMLResponse")
def index(request: Request):
    context={'request':request}
    return templates.TemplateResponse("index.html",context)


@app.post("/submitted")
async def create_file(image_file: UploadFile = File(...)):
    image_data=await image_file.read()
    #print(image_data)
    bgr_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    #bgr_image = cv2.imread(bgr_image)
    print(type(bgr_image))
    print(bgr_image.shape)
    value =deploy_and_test(bgr_image)
    return JSONResponse (content={"message": value}, status_code=200) 





    