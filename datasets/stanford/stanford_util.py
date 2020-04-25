import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from config import STANFORD_CARS_CARS_META,STANFORD_CARS_DOWNSAMPLE_SUFFIX

# Read ain .mat file
def load_anno(path):
    mat = scipy.io.loadmat(path)
    return mat

def load_class_names(path=STANFORD_CARS_CARS_META):
    cn = load_anno(path)['class_names']
    cn = cn.tolist()[0]
    cn = [str(c[0].item()) for c in cn]
    return cn

def load_annotations(path,downsample):
    ann = load_anno(path)['annotations'][0]
    ret = {}

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        # Use downsamples stanford image if passed in
        if downsample:
            fn = imgfn.item()[:-4] + STANFORD_CARS_DOWNSAMPLE_SUFFIX
        else:
            fn = imgfn.item()

        r = {
            'x1': x1.item(),
            'y1': y1.item(),
            'x2': x2.item(),
            'y2': y2.item(),
            'target': target.item() - 1,
            'filename': fn
        }
        ret[idx] = r
    return ret