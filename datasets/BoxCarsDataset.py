import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np

from config import BOXCARS_DATASET_ROOT,BOXCARS_IMAGES_IMAGES,BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES

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
    def __init__(self, imgdir, anno_path, transform, size,split,part):
        boxCarsAnnUtil = BoxCarDataSetUtil(split,part)
        self.annos  = boxCarsAnnUtil.load_annotations_boxcars()
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
        

            return vehicle, instance, 

    def load_annotations_boxcars(self):
        
        self.get_class_names()
        self.initialize_data()

        ret = {}
        img_counter = 0
        for car_ids in self.X[self.current_part]:
            vehicle, instance = self.get_vehicle_instance_data(car_ids[0], car_ids[1])
            r = {
                'x1': None,
                'y1': None,
                'x2':None,
                'y2': None,
                'target': self.convert_ann_to_num(vehicle['annotation']),
                'filename': instance['path']
            }
            ret[img_counter] = r
            img_counter=img_counter+1  
            
        return ret

    def convert_ann_to_num(self,ann):
        if ann not in self.cars_annotations:
            print("Ann not there. Error occured")
            exit(1)
            self.cars_annotations.append(ann)
        return self.cars_annotations.index(ann)

    def get_class_names(self):
        with open(BOXCARS_HARD_CLASS_NAMES, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                self.cars_annotations.append(currentPlace)
