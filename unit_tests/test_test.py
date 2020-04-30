import sys
import os
from os import path

if ('..' not in sys.path) : sys.path.append("..")
from train import main as main_train
from tests import main as main_test


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    if os.getcwd().split('/')[-1].lower() != 'lowresimagerecognision' : os.chdir("..")
    print('after',os.getcwd())

def teardown_method(self, method):
    """ teardown any state that was previously setup with a setup_method
    call.
    """
    os.remove("saves/1000000000_history.csv")
    os.remove("saves/1000000000_model.pth")


def test_test_model():

    config = {
        'batch_size': 1,
        'test_batch_size': 1,
        'epochs': 1,
        'imgsize': (224,224),
        'model_version': 1,
        'train_test_version': 1,
        'dataset_version': 4,
        'boxcar_split': 'hard',
        'train_samples':1,
        'train_samples_percent':True,

        'finetune': False,
        'model_id': 1000000000,
        'finetune_stan_box': False,
        'fine-tune-id':1,
        'ds-stanford':False,
        'loss-function':1,
        
        'lr': 1e-4,
        'weight_decay': 0.9,
        'momentum': 0.8,
        'adam': True,
        
        'main_loss': 1,
        'make_loss': 0.2,
        'model_loss': 0.2,
        'submodel_loss': 0.2,
        'generation_loss': 0.2,
        'testing':False
    }
    #Get a model
    main_train(config)
    #Evalute the model
    main_test(config)

    assert path.exists("saves/" +str(config['model_id'])+ "_history.csv")
    assert path.exists("saves/" +str(config['model_id'])+ "model.pth")



