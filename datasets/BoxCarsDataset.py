import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import random
from config import BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES,BOXCARS_HARD_MAKE_NAMES,BOXCARS_HARD_MODEL_NAMES,BOXCARS_HARD_SUBMODEL_NAMES,BOXCARS_HARD_GENERATION_NAMES
from datasets.boxcars_image_transformations import alter_HSV, image_drop, add_bb_noise_flip,unpack_3DBB
from collections import Counter
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
                img, r['bb3d'] = add_bb_noise_flip(img, r['bb3d'], flip, bb_noise) 

            img = unpack_3DBB(img, r['bb3d']) 
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
        y_categorical = np.zeros((y.shape[0],self.get_number_of_classes()['total']))
        y_categorical[np.arange(y.shape[0]), y] = 1
        self.Y[self.current_part] = y_categorical

    def get_number_of_classes(self):
        class_sizes = {
            'total':len(self.split["types_mapping"]),
            'make':len(self.make),
            'model':len(self.model),
            'submodel':len(self.submodel),
            'generation':len(self.generation),
        }

        return class_sizes

    def get_vehicle_instance_data(self,vehicle_id, instance_id, original_image_coordinates=False):
            """
            original_image_coordinates: the 3DBB coordinates are in the original image space
                                        to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                        which is done if this parameter is False. 
            """
            vehicle = self.dataset["samples"][vehicle_id]
            instance = vehicle["instances"][instance_id]
            bb3d = instance["3DBB"]
            bb3d = bb3d - instance["3DBB_offset"]

            return vehicle, instance, bb3d

    def load_annotations_boxcars_v1(self):
        
        self.cars_annotations = self.get_class_names(BOXCARS_HARD_CLASS_NAMES)
        self.initialize_data()
        ret = {}
        img_counter = 0
        for car_ids in self.X[self.current_part]:
            vehicle, instance,bb3d = self.get_vehicle_instance_data(car_ids[0], car_ids[1])
            r = {
                'target': self.get_array_index_of_string(self.cars_annotations,vehicle['annotation']),
                'filename': instance['path']
            }
            ret[img_counter] = r
            img_counter=img_counter+1  
            
        return ret

    def load_annotations_boxcars_v2(self):
        
        self.cars_annotations = self.get_class_names(BOXCARS_HARD_CLASS_NAMES)
        self.initialize_data()

        self.make = self.get_class_names(BOXCARS_HARD_MAKE_NAMES)
        self.model = self.get_class_names(BOXCARS_HARD_MODEL_NAMES)
        self.submodel = self.get_class_names(BOXCARS_HARD_SUBMODEL_NAMES)
        self.generation = self.get_class_names(BOXCARS_HARD_GENERATION_NAMES)
        
        # count_cars_annotations = { i : 0 for i in self.cars_annotations }
        
        ret = {}
        img_counter = 0
        for car_ids in self.X[self.current_part]:
            vehicle, instance, bb3d = self.get_vehicle_instance_data(car_ids[0], car_ids[1])
            make,model,submodel,generation = vehicle['annotation'].split()
            r = {
                'target': self.get_array_index_of_string(self.cars_annotations, vehicle['annotation']),  
                'make':self.get_array_index_of_string(self.make, make),
                'model':self.get_array_index_of_string(self.model, model),
                'submodel':self.get_array_index_of_string(self.submodel, submodel),
                'generation':  self.get_array_index_of_string(self.generation, generation),    
                'filename': instance['path'],
                'bb3d':bb3d
            }
            # count_cars_annotations[vehicle['annotation']]+=1
            ret[img_counter] = r
            img_counter=img_counter+1  

        return ret
    
    def get_array_index_of_string(self,arr,ann):
        if ann not in arr:
            print("Ann not in array. Check this. \nError occured",ann,arr)
            exit(1)
            # arr.append(ann)
        return arr.index(ann)

    def get_class_names(self,filename):
        ann = []
        with open(filename, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                ann.append(currentPlace)
        return ann
    

    def reduced_dataset(self,num_train_samples,full_annos,num_train_samples_percent):
        
        
        #Get current list of car_annocations and convert to counter list
        temp_cars_annotations = self.cars_annotations

        temp_cars_annotations = {'skoda fabia combi mk1': 4056,'skoda fabia hatchback mk2': 1383, 'skoda octavia combi mk2': 4298, 'skoda fabia combi mk2': 3087, 'skoda octavia sedan mk3': 1159, 'skoda octavia combi mk3': 794, 'skoda octavia combi mk1': 2751, 'volkswagen passat combi mk6': 341, 'skoda fabia hatchback mk1': 2947, 'skoda superb sedan mk2': 198, 'skoda rapid sedan mk1': 273, 'skoda octavia sedan mk1': 2738, 'skoda octavia sedan mk2': 2185, 'skoda superb combi mk2': 239, 'volkswagen passat combi mk7': 88, 'volkswagen golf hatchback mk6': 178, 'volkswagen passat sedan mk6': 227, 'volkswagen passat sedan mk5': 89, 'skoda superb sedan mk1': 187, 'volkswagen passat combi mk5': 668, 'skoda fabia sedan mk1': 189, 'skoda yeti suv mk1': 592, 'volkswagen caddy van mk3': 538, 'skoda citigo hatchback mk1': 1157, 'ford focus combi mk1': 807, 'peugeot 206 hatchback mk1': 569, 'skoda felicia combi mk2': 265, 'skoda felicia hatchback mk1': 925, 'fiat panda hatchback mk2': 336, 'citroen berlingo van mk2': 495, 'bmw x5 suv mk2': 121, 'audi a6 combi mk4': 101, 'hyundai i20 hatchback mk1': 290, 'bmw x3 suv mk2': 133, 'porsche cayenne suv mk2': 60, 'citroen berlingo van mk1': 597, 'ford fiesta hatchback mk6': 145, 'skoda felicia hatchback mk2': 958, 'peugeot partner van mk2': 218, 'skoda favorit hatchback mk1': 195, 'audi a6 sedan mk4': 294, 'toyota auris combi mk2': 121, 'toyota yaris hatchback mk3': 228, 'hyundai ix20 mpv mk1': 420, 'renault megane combi mk3': 381, 'opel corsa hatchback mk4': 164, 'ford mondeo combi mk3': 242, 'seat alhambra mpv mk1': 69, 'skoda felicia combi mk1': 117, 'bmw 1 hatchback mk1': 75, 'ford focus combi mk3': 140, 'volkswagen golf combi mk4': 105, 'volkswagen touareg suv mk2': 171, 'volkswagen sharan mpv mk1': 401, 'kia ceed combi mk1': 417, 'fiat punto hatchback mk1': 190, 'audi q7 suv mk1': 135, 'renault kangoo van mk1': 141, 'volkswagen golf hatchback mk4': 523, 'volkswagen tiguan suv mk1': 86, 'hyundai i30 hatchback mk1': 133, 'ford focus combi mk2': 312, 'opel corsa hatchback mk3': 175, 'hyundai i30 combi mk2': 381, 'skoda roomster van mk1': 1206, 'peugeot 107 hatchback mk1': 223, 'opel astra combi mk2': 171, 'renault megane combi mk2': 251, 'citroen c1 hatchback mk1': 293, 'hyundai i30 hatchback mk2': 211, 'renault clio hatchback mk2': 315, 'renault trafic van mk2': 388, 'ford s-max mpv mk1': 223, 'seat cordoba sedan mk2': 112, 'seat leon hatchback mk2': 109, 'ford fusion mpv mk1': 550, 'renault laguna combi mk2': 94, 'volkswagen touran mpv mk1': 254, 'ford focus hatchback mk1': 329, 'fiat punto hatchback mk2': 141, 'peugeot 207 hatchback mk1': 76, 'renault megane hatchback mk3': 124, 'ford focus hatchback mk2': 194, 'renault thalia sedan mk1': 145, 'hyundai getz hatchback mk1': 125, 'renault scenic mpv mk1': 121, 'volvo xc90 suv mk1': 122, 'opel vivaro van mk2': 134, 'ford transit van mk6': 438, 'peugeot partner van mk1': 175, 'volkswagen transporter van mk5': 417, 'kia sportage suv mk3': 143, 'renault master van mk2': 153, 'citroen c3 hatchback mk1': 187, 'volvo xc60 suv mk1': 114, 'volkswagen transporter van mk4': 383, 'ford mondeo combi mk4': 157, 'peugeot 308 combi mk1': 81, 'renault kangoo van mk2': 145, 'peugeot boxer van mk3': 117, 'peugeot 307 combi mk1': 161, 'ford transit van mk7': 265, 'fiat ducato van mk3': 192, 'hyundai ix35 suv mk1': 178, 'fiat doblo van mk1': 163, 'skoda fabia hatchback mk3': 139, 'renault master van mk3': 174}

        #Distinct samples from each
        if(not num_train_samples_percent):
            for x in temp_cars_annotations:
                if temp_cars_annotations[x] > num_train_samples:
                    temp_cars_annotations[x] = num_train_samples
        else:#Percentage from each class
            if(num_train_samples<=0 or num_train_samples > 1):
                print("The percentage number must be between 1 and 100")
                exit()
            for x in temp_cars_annotations:    
                temp_cars_annotations[x] = round(temp_cars_annotations[x]*num_train_samples)
        ret = {}

        car_ann_counter = 0
        img_counter = 0
        #Run through each annoation

        while not len(temp_cars_annotations) is 0:
            target_ann = self.cars_annotations[full_annos[car_ann_counter]['target']]
            if target_ann in temp_cars_annotations:
                ret[img_counter] = full_annos[car_ann_counter]
                img_counter+=1
                temp_cars_annotations[target_ann]-=1
                if temp_cars_annotations[target_ann] == 0:
                    temp_cars_annotations.pop(target_ann)               


            car_ann_counter+=1  

        return ret    