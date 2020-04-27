import torchvision
import sys
import os
sys.path.append("..")
from config import BOXCARS_IMAGES_IMAGES,BOXCARS_HARD_MAKE_NAMES
from datasets.boxcars.boxcars_datasets import BoxCarsDatasetV1,BoxCarsDatasetV1_2,BoxCarsDatasetV2,BoxCarsDatasetV3
from datasets.boxcars.boxcars_datasets import BoxCarDataSetUtil
import numpy as np
from torchvision import transforms
import torch

os.chdir("..")

def test_load_util_object():
    boxcars_util = BoxCarDataSetUtil('hard','train')

    assert boxcars_util is not None


def test_get_number_of_classes():
    boxcars_util = BoxCarDataSetUtil('hard','train')
    class_sizes = boxcars_util.get_number_of_classes()
    assert class_sizes['total'] ==107
    assert class_sizes['make'] == 0
    assert class_sizes ['model'] == 0
    assert class_sizes['submodel'] == 0
    assert class_sizes['generation'] == 0

    #Process labels to create the mulitask labels
    boxcars_util.load_annotations_boxcars_v2()
    class_sizes = boxcars_util.get_number_of_classes()

    assert class_sizes['total'] ==107
    assert class_sizes['make'] == 16
    assert class_sizes ['model'] == 68
    assert class_sizes['submodel'] == 6
    assert class_sizes['generation'] == 7

def test_get_vehicle_instance_data():
    boxcars_util = BoxCarDataSetUtil('hard','train')
    vehicle, instance, bb3d = boxcars_util.get_vehicle_instance_data(0,0)


    assert vehicle is not None
    assert instance is not None
    assert bb3d is not None

def test_load_annotations_boxcars_v1():
    boxcars_util = BoxCarDataSetUtil('hard','train')

    ann = boxcars_util.load_annotations_boxcars_v1()

    assert ann is not None
    assert type(ann[0]['target']) is int
    assert type(ann[0]['filename']) is str


def test_load_annotations_boxcars_v2():
    boxcars_util = BoxCarDataSetUtil('hard','train')

    ann = boxcars_util.load_annotations_boxcars_v2()

    assert ann is not None
    assert type(ann[0]['target']) is int
    assert type(ann[0]['filename']) is str
    assert type(ann[0]['make']) is int
    assert type(ann[0]['model']) is int
    assert type(ann[0]['submodel']) is int
    assert type(ann[0]['generation']) is int
    assert type(ann[0]['bb3d']) is np.ndarray


    class_sizes = boxcars_util.get_number_of_classes()

    assert class_sizes['total'] ==107
    assert class_sizes['make'] == 16
    assert class_sizes ['model'] == 68
    assert class_sizes['submodel'] == 6
    assert class_sizes['generation'] == 7



def test_get_array_index_of_string():
    boxcars_util = BoxCarDataSetUtil('hard','train')
    ann = boxcars_util.get_class_names(BOXCARS_HARD_MAKE_NAMES)

    assert boxcars_util.get_array_index_of_string(ann,'fiat') == 0
    assert boxcars_util.get_array_index_of_string(ann,'volkswagen') == 2
    assert boxcars_util.get_array_index_of_string(ann,'hyundai') == 15

def test_get_class_names():
    boxcars_util = BoxCarDataSetUtil('hard','train')
    ann = boxcars_util.get_class_names(BOXCARS_HARD_MAKE_NAMES)

    assert len(ann) == 16
    assert boxcars_util.get_array_index_of_string(ann,'fiat') == 0
    assert boxcars_util.get_array_index_of_string(ann,'volkswagen') == 2
    assert boxcars_util.get_array_index_of_string(ann,'hyundai') == 15


def test_reduced_dataset():
    boxcars_util = BoxCarDataSetUtil('hard','train')
    ann = boxcars_util.load_annotations_boxcars_v1()

    assert len(boxcars_util.reduced_dataset(1,ann,False)) == 107
    assert len(boxcars_util.reduced_dataset(10,ann,False)) == 1070
    assert len(boxcars_util.reduced_dataset(1,ann,True)) == 51691



