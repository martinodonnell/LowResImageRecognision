import torchvision
import sys
import os
if ('..' not in sys.path) : sys.path.append("..")
from config import STANFORD_CARS_TEST_ANNOS,BOXCARS_HARD_CLASS_NAMES,STANFORD_CARS_TRAIN_ANNOS
from datasets.stanford.stanford_util import load_anno,load_class_names,load_annotations

from torchvision import transforms
import torch

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    if os.getcwd().split('/')[-1].lower() != 'lowresimagerecognision' : os.chdir("..")
    print('after',os.getcwd())
    
def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    print("os", os.getcwd())

def test_load_anno():
    ann = load_anno(STANFORD_CARS_TEST_ANNOS)
    print(ann)
    assert len(ann) > 0


def test_load_class_names():
    class_names = load_class_names()
    assert len(class_names) == 196

def test_load_annotations():
    ann = load_annotations(STANFORD_CARS_TRAIN_ANNOS,False)
    assert len(ann) ==8144

    ann = load_annotations(STANFORD_CARS_TRAIN_ANNOS,True)
    assert len(ann) ==8144
