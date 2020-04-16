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

class CarsDatasetV1(Dataset):
    def __init__(self, imgdir, anno_path, transform, size,downsample):        
        self.annos = load_annotations(anno_path,downsample)     
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target


def load_annotations_v2(path, v2_info,downsample):
    ann = load_anno(path)['annotations'][0]
    ret = {}
    make_codes = v2_info['make'].astype('category').cat.codes
    type_codes = v2_info['model_type'].astype('category').cat.codes

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
            'make_target': make_codes[target.item() - 1].item(),
            'type_target': type_codes[target.item() - 1].item(),
            'filename': fn
        }

        ret[idx] = r
    return ret


def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'

        if model == 'Integra Type R':
            model_type = 'Type-R'

        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'

        if 'SRT' in model_type:
            model_type = 'SRT'

        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pd.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr

class CarsDatasetV2(Dataset):
    def __init__(self, imgdir, anno_path, transform, size,downsample):
        self.class_names = load_class_names()
        self.v2_info = separate_class(self.class_names)
        self.annos = load_annotations_v2(anno_path, self.v2_info,downsample)
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}
      

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]

        target = r['target']
        make_target = r['make_target']
        type_target = r['type_target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)
            
            self.cache[idx] = img
        else:
            img = self.cache[idx]
        
        img = self.transform(img)

        return img, target, make_target, type_target


#Mixed dataset
class CarsDatasetV3(Dataset):
    def __init__(self, imgdir, anno_path, transform, size,downsample):        
        self.annos = load_annotations(anno_path,False)
        self.annos_downsample = load_annotations(anno_path,True)

        next_key = len(self.annos)
        for value in self.annos_downsample.values():
            self.annos[next_key] = value
            next_key+=1

        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]
        target = r['target']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)

        return img, target

