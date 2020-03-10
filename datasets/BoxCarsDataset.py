import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import random
from config import BOXCARS_DATASET_ROOT,BOXCARS_IMAGES_IMAGES,BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES
from datasets.boxcars_image_transformations import alter_HSV, image_drop, add_bb_noise_flip,unpack_3DBB

def load_boxcar_class_names():
    ann = []
    with open(BOXCARS_HARD_CLASS_NAMES, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                ann.append(currentPlace)

    return ann

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
            #_______

            # #Added to see if this improves on the base
            # img = alter_HSV(img) # randomly alternate color
            # img = image_drop(img) # randomly remove part of the image

            #_______
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
        make_target = r['make']
        model_target = r['model']
        submodel_target = r['submodel']
        generation_target =r['generation']

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
                img, bb3d = add_bb_noise_flip(img, r['bb3d'], flip, bb_noise) 

            img = unpack_3DBB(img, bb3d) 
            # img = (img.astype(np.float32) - 116)/128.

            img = Image.fromarray(img)
            self.cache[idx] = img
        else:
            img = self.cache[idx]

        img = self.transform(img)
        return img, target



#Boxcar stuff start
class BoxCarDataSetUtil(object):
    def __init__(self,new_split,new_part):
        self.split = self.load_cache(BOXCARS_CLASSIFICATION_SPLITS)[new_split]
        self.dataset = self.load_cache(BOXCARS_DATASET)
        self.current_part = new_part
        self.X = {}
        self.Y = {}
        self.X[new_part] = None
        self.Y[new_part] = None # for labels as array of 0-1 flags
        self.cars_annotations = []
        self.v2_info = []
        self.make = []
        self.model = []
        self.submodel = []
        self.generation = []
        

    def load_cache(self,path, encoding="latin-1", fix_imports=True):
        """
        encoding latin-1 is default for Python2 compatibility
        """
        with open(path, "rb") as f:
            return pickle.load(f, encoding=encoding, fix_imports=True)

    def initialize_data(self):
        data = self.split[self.current_part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label]*num_instances)
        self.X[self.current_part] = np.asarray(x,dtype=int)

        y = np.asarray(y,dtype=int)
        y_categorical = np.zeros((y.shape[0],self.get_number_of_classes()))
        y_categorical[np.arange(y.shape[0]), y] = 1
        self.Y[self.current_part] = y_categorical

    def get_number_of_classes(self):
        return len(self.split["types_mapping"])

    def get_vehicle_instance_data(self,vehicle_id, instance_id, original_image_coordinates=False):
            """
            original_image_coordinates: the 3DBB coordinates are in the original image space
                                        to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                        which is done if this parameter is False. 
            """
            vehicle = self.dataset["samples"][vehicle_id]
            instance = vehicle["instances"][instance_id]
            bb3d = instance["3DBB"]

            return vehicle, instance, bb3d

    def load_annotations_boxcars_v1(self):
        
        self.get_class_names()
        self.initialize_data()
        ret = {}
        img_counter = 0
        for car_ids in self.X[self.current_part]:
            vehicle, instance,bb3d = self.get_vehicle_instance_data(car_ids[0], car_ids[1])
            r = {
                'x1': None,
                'y1': None,
                'x2':None,
                'y2': None,
                'target': self.get_array_index_of_string(self.cars_annotations,vehicle['annotation']),
                'filename': instance['path']
            }
            ret[img_counter] = r
            img_counter=img_counter+1  
            
        return ret

    def load_annotations_boxcars_v2(self):
        
        self.get_class_names()
        self.initialize_data()
        self.separate_classes()
        
        ret = {}
        img_counter = 0
        for car_ids in self.X[self.current_part]:
            vehicle, instance, bb3d = self.get_vehicle_instance_data(car_ids[0], car_ids[1])
            make,model,submodel,generation = vehicle['annotation'].split()
            r = {
                'x1': None,
                'y1': None,
                'x2':None,
                'y2': None,
                'target': self.get_array_index_of_string(self.cars_annotations, vehicle['annotation']),  
                'make':self.get_array_index_of_string(self.make, make),
                'model':self.get_array_index_of_string(self.model, model),
                'submodel':self.get_array_index_of_string(self.submodel, submodel),
                'generation':  self.get_array_index_of_string(self.generation, generation),    
                'filename': instance['path'],
                'bb3d':bb3d
            }
            ret[img_counter] = r
            img_counter=img_counter+1  
        return ret

    def separate_classes(self):     
        for data in self.cars_annotations:
            t_make,t_model,t_submodel,t_generation = data.split()
            self.make.append(t_make)
            self.model.append(t_model)
            self.submodel.append(t_submodel)
            self.generation.append(t_generation)

        self.make = list(set(self.make))
        self.model = list(set(self.model))
        self.submodel = list(set(self.submodel))
        self.generation = list(set(self.generation))


    
    def get_array_index_of_string(self,arr,ann):
        if ann not in arr:
            print("Ann not in array. Check this. \nError occured",ann,arr)
            exit(1)
            arr.append(ann)
        return arr.index(ann)

    def get_class_names(self):
        with open(BOXCARS_HARD_CLASS_NAMES, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                self.cars_annotations.append(currentPlace)