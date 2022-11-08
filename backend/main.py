from fastapi import FastAPI,File,Form,UploadFile, Response
import cv2
import numpy as np
import scipy
from PIL import Image

app=FastAPI()

def pencilsketch(pixels):##this takes a numpy array as input 
    img=cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray###returns a numpy array

@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.post("/uploadimage/")
async def createimage(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=pencilsketch(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")


