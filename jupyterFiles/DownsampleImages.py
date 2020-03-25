#Down samples images from stanford to (122,122) to match that of boxcars
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

folder="/Users/martinodonnell/Documents/uni/fourth_year/4006/LowResImageRecognision/data/StanfordCars/"
for subdir, dirs, files in os.walk(folder):
    for file in tqdm(files):
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):
            img = cv2.imread(filepath)
            res = cv2.resize(img, dsize=(122,122), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(filepath[:-4] + "_ds.jpg", res)