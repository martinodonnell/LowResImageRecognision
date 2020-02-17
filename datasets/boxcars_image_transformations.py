# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
from PIL import Image
def alter_HSV(img, change_probability = 0.6):
    if random.random() < 1-change_probability:
        return img

    #Convert from pil to cv
    img = np.array(img) 
    addToHue = random.randint(0,179)
    addToSaturation = random.gauss(60, 20)
    addToValue = random.randint(-50,50)
    hsvVersion =  cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    channels = hsvVersion.transpose(2, 0, 1)
    channels[0] = ((channels[0].astype(int) + addToHue)%180).astype(np.uint8)
    channels[1] = (np.maximum(0, np.minimum(255, (channels[1].astype(int) + addToSaturation)))).astype(np.uint8)
    channels[2] = (np.maximum(0, np.minimum(255, (channels[2].astype(int) + addToValue)))).astype(np.uint8)
    hsvVersion = channels.transpose(1,2,0)   
        
    return Image.fromarray(cv2.cvtColor(hsvVersion, cv2.COLOR_HSV2RGB))

def image_drop(img, change_probability = 0.6):
    if random.random() < 1-change_probability:
        return img
    #Convert from pil to cv
    img = np.array(img) 
    width = random.randint(int(img.shape[1]*0.10), int(img.shape[1]*0.3))
    height = random.randint(int(img.shape[0]*0.10), int(img.shape[0]*0.3))
    x = random.randint(int(img.shape[1]*0.10), img.shape[1]-width-int(img.shape[1]*0.10))
    y = random.randint(int(img.shape[0]*0.10), img.shape[0]-height-int(img.shape[0]*0.10))
    img[y:y+height,x:x+width,:] = (np.random.rand(height,width,3)*255).astype(np.uint8)
    return Image.fromarray(img)