import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np

# Read ain .mat file
def load_anno(path):
    mat = scipy.io.loadmat(path)
    return mat

def load_class_names(path='data/StanfordCars/devkit/cars_meta.mat'):
    cn = load_anno(path)['class_names']
    cn = cn.tolist()[0]
    cn = [str(c[0].item()) for c in cn]
    return cn

def load_annotations(path):
    ann = load_anno(path)['annotations'][0]
    ret = {}

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        r = {
            'x1': x1.item(),
            'y1': y1.item(),
            'x2': x2.item(),
            'y2': y2.item(),
            'target': target.item() - 1,
            'filename': imgfn.item()
        }

        ret[idx] = r

    return ret

class CarsDataset(Dataset):
    def __init__(self, imgdir, anno_path, transform, size,dataset,split,part):
        if(dataset==1):
            self.annos = load_annotations(anno_path)
        elif(dataset==2):
            boxCarsAnnUtil = BoxCarDataset(imgdir, anno_path, transform, size,dataset,split,part)
            self.annos  = boxCarsAnnUtil.load_annotations_boxcars()
        else:
            print("No dataset. Leaving")
            exit(1)

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
class BoxCarDataset(object):
    def __init__(self, imgdir, anno_path, transform, size,dataset,new_split,new_part):
        self.split = self.load_cache(anno_path + "/classification_splits.pkl")[new_split]
        self.dataset = self.load_cache(anno_path +  "/dataset.pkl")
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
        
        self.create_unique_ann_list()
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

    def create_unique_ann_list(self):
        
        for data in self.dataset['samples']:
            new_ann = data['annotation']
            if new_ann not in self.cars_annotations:
                self.cars_annotations.append(new_ann)

    def convert_ann_to_num(self,ann):
        return self.cars_annotations.index(ann)

    #Boxcar stuff end

def prepare_loader(config):
    if(config['dataset']==1):
        train_imgdir = 'data/cars_train'
        test_imgdir = 'data/cars_test'

        train_annopath = 'data/devkit/cars_train_annos.mat'
        test_annopath = 'data/devkit/cars_test_annos_withlabels.mat'

    elif(config['dataset']==2):
        train_imgdir = test_imgdir =  'data/BoxCars/images'
        train_annopath = test_annopath = 'data/BoxCars'
    else:
        print("No dataset. Leaving")
        exit(1)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    
    train_dataset = CarsDataset(train_imgdir, train_annopath, train_transform, config['imgsize'],config['dataset'],config['split'],'train')
    test_dataset = CarsDataset(test_imgdir, test_annopath, test_transform, config['imgsize'],config['dataset'],config['split'],'test')

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=False,
                              num_workers=12)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return train_loader, test_loader

    

