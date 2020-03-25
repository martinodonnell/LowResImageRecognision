# -*- coding: utf-8 -*-
import os
#%%
# change this to your location
# BOXCARS_DATASET_ROOT = "/mnt/scratch/users/40160005/BoxCars/"
BOXCARS_DATASET_ROOT = "data/BoxCars" 
STANFORD_DATASET_ROOT = "data/StanfordCars" 
SAVE_FOLDER = 'saves'
CONFUSION_MATRIX = os.path.join(SAVE_FOLDER, "confusionMatrix")



#%%
BOXCARS_IMAGES_IMAGES = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")
BOXCARS_HARD_CLASS_NAMES = "data/BoxCars/classNames/hard_class_names.txt"
BOXCARS_HARD_MAKE_NAMES = "data/BoxCars/classNames/hard_make_names.txt"
BOXCARS_HARD_MODEL_NAMES = "data/BoxCars/classNames/hard_model_names.txt"
BOXCARS_HARD_SUBMODEL_NAMES = "data/BoxCars/classNames/hard_submodel_names.txt"
BOXCARS_HARD_GENERATION_NAMES = "data/BoxCars/classNames/hard_generation_names.txt"

STANFORD_CARS_TRAIN = os.path.join(STANFORD_DATASET_ROOT, 'cars_train')
STANFORD_CARS_TEST = os.path.join(STANFORD_DATASET_ROOT, 'cars_test')
STANFORD_CARS_TRAIN_ANNOS = os.path.join(STANFORD_DATASET_ROOT, 'devkit/cars_train_annos.mat')
STANFORD_CARS_TEST_ANNOS = os.path.join(STANFORD_DATASET_ROOT, 'devkit/cars_test_annos_withlabels.mat')
STANFORD_CARS_CARS_META = os.path.join(STANFORD_DATASET_ROOT, 'devkit/cars_meta.mat')
STANFORD_CARS_DOWNSAMPLE_SUFFIX = "_ds.jpg"
