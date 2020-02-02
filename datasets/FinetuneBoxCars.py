import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from config import BOXCARS_DATASET_ROOT,BOXCARS_IMAGES_IMAGES,BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES


def changeAnnToClassNames(ann,classNames):
    print("classNames",classNames)
    print("ann :",ann)
