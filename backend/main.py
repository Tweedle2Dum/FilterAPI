from fastapi import FastAPI,File,Form,UploadFile, Response
import cv2
import numpy as np
import scipy
from PIL import Image
from scipy.interpolate import UnivariateSpline

def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

app=FastAPI()

def pencilsketch(pixels):##this takes a numpy array as input 
    img=cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray###returns a numpy array

def greyscale(pixels):
    img = cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return greyscale

def sepia(pixels):
    img = cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

def invert(pixels):
    img=cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    inv = cv2.bitwise_not(img)
    return inv
def summer(pixels):
    img=cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

def Winter(pixels):
    img=cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.post("/uploadimage/pencilsketch")
async def createpencilsketch(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=pencilsketch(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")

@app.post("/uploadimage/greyscale")
async def creategreyscale(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=greyscale(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")


@app.post("/uploadimage/sepia")
async def createsepia(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=sepia(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")


@app.post("/uploadimage/invert")
async def createinvert(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=invert(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")

@app.post("/uploadimage/summereffect")
async def createsummereffect(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=summer(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")

@app.post("/uploadimage/winter")
async def createwintereffect(img:UploadFile):
    #img=Image.open(img.file)
    pixels=np.array(bytearray(img.file.read()))
    x=Winter(pixels)
    x,y=cv2.imencode(".png",x)
    return Response(y.tobytes(),media_type="image/png")

