import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

from datasets.boxcars.boxcars_util import BoxCarDataSetUtil
from datasets.boxcars.boxcars_image_transformations import alter_HSV, image_drop, unpack_3DBB, add_bb_noise_flip

class BoxCarsDatasetV1(Dataset):
    def __init__(self, imgdir, transform, size,split,part):
        boxCarsAnnUtil = BoxCarDataSetUtil(split,part)
        self.annos  = boxCarsAnnUtil.load_annotations_boxcars_v1()
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


class BoxCarsDatasetV1_2(Dataset):
    def __init__(self, imgdir, transform, size,split,part,num_train_samples,num_train_samples_percent):
        boxCarsAnnUtil = BoxCarDataSetUtil(split,part)
        self.annos  = boxCarsAnnUtil.load_annotations_boxcars_v1()
        if part == 'train':
            self.annos = boxCarsAnnUtil.reduced_dataset(num_train_samples,self.annos,num_train_samples_percent)

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


class BoxCarsDatasetV2(Dataset):
    def __init__(self, imgdir, transform, size,split,part):
        boxCarsAnnUtil = BoxCarDataSetUtil(split,part)
        self.annos  = boxCarsAnnUtil.load_annotations_boxcars_v2()
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]
        target = r['target']
        make_target = r['make']
        model_target = r['model']
        submodel_target = r['submodel']
        generation_target =r['generation']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            img = img.convert('RGB')
            img = self.resize(img)

            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)
        return img, target,make_target,model_target,submodel_target,generation_target



class BoxCarsDatasetV3(Dataset):
    def __init__(self, imgdir, transform, size,split,part):
        boxCarsAnnUtil = BoxCarDataSetUtil(split,part)
        self.part = part
        self.annos  = boxCarsAnnUtil.load_annotations_boxcars_v2()
        self.imgdir = imgdir
        self.transform = transform
        self.resize = transforms.Resize(size)
        self.cache = {}

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        r = self.annos[idx]
        target = r['target']
        # make_target = r['make']
        # model_target = r['model']
        # submodel_target = r['submodel']
        # generation_target =r['generation']

        if idx not in self.cache:
            fn = r['filename']

            img = Image.open(os.path.join(self.imgdir, fn))
            #Convert from pil to cv
            img = np.array(img)

            if self.part=='train':
                img = alter_HSV(img) # randomly alternate color
                img = image_drop(img) # randomly remove part of the image
                bb_noise = np.clip(np.random.randn(2) * 1.5, -5, 5) # generate random bounding box movement
                flip = bool(random.getrandbits(1)) # random flip
                img, r['bb3d'] = add_bb_noise_flip(img, r['bb3d'], flip, bb_noise) 

            img = unpack_3DBB(img, r['bb3d']) 
            # img = (img.astype(np.float32) - 116)/128.

            img = Image.fromarray(img)
            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)
        return img, target