from PIL import Image
import cv2
import numpy as np
import random
from datasets.boxcars_image_transformations import alter_HSV, image_drop
from PIL import ImageFilter
from os.path import getsize, isfile, isdir, join


backupname = '/Users/martinodonnell/Desktop/019700_000.png'
filename = '/Users/martinodonnell/Desktop/019700_000_backup.png'
img = Image.open(backupname)
print(img.size)


temp = img.fromArray
test2 = img.resize((224,224))


test2.show()

# print(img.info['dpi'])
# img.save(filename, quality=10)

# test = Image.open(backupname)
# test.show()
# print(test.info['dpi'])

# origsize = getsize(backupname)
# newsize = getsize(filename)

# print(origsize,newsize)
