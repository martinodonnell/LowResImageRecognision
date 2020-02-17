from PIL import Image
import cv2
import numpy as np
import random
from datasets.boxcars_image_transformations import alter_HSV, image_drop

img = Image.open('data/BoxCars/images/012/12/022162_009.png')
image_drop(img).show()

