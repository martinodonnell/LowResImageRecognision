import sys
sys.path.append("..")
from train_test_util import get_output_filepaths,set_up_output_filepaths,load_weight,get_args,dual_cross_entropy,get_loss_function,load_weight_stanford
from exceptions.exceptions import OutputFileExistsException,InValidTestTrainMethod,InvalidLossFunction
from models import construct_model
import torch
import pytest
import os
from config import SAVE_FOLDER
import torch.nn.functional as F

def test_get_output_filepaths():
    id_string = "111"
    csv_history_filepath, model_best_filepath = get_output_filepaths(id_string)
    assert csv_history_filepath == os.path.join(SAVE_FOLDER, id_string + '_history.csv')
    assert model_best_filepath == os.path.join(SAVE_FOLDER, id_string + '_model.pth')


def test_set_up_output_filepaths_test_mode():
    testing=True
    id_string = "1234"
    train_test_method = 1

    csv_history_filepath, model_best_filepath = set_up_output_filepaths(testing,id_string,train_test_method)
    assert csv_history_filepath == "Testing_csv"
    assert model_best_filepath == "Testing_Model"

    #Files should not exist because test mode is on
    assert os.path.isfile(csv_history_filepath) == False
    assert os.path.isfile(model_best_filepath) == False

def test_set_up_output_filepaths_create_save_files():

    testing=False
    id_string = "1234"
    train_test_method = 1

    os.chdir('../')

    csv_history_filepath, model_best_filepath = set_up_output_filepaths(testing,id_string,train_test_method)
    assert csv_history_filepath == os.path.join(SAVE_FOLDER, id_string + '_history.csv')
    assert model_best_filepath == os.path.join(SAVE_FOLDER, id_string + '_model.pth')

    #Files should be created
    assert os.path.isfile(csv_history_filepath) == True

    #Clean up 
    os.remove(csv_history_filepath)


def test_set_up_output_filepaths_already_exists():

    testing=False
    id_string = "1234"
    train_test_method = 1


    csv_history_filepath, model_best_filepath = set_up_output_filepaths(testing,id_string,train_test_method)
    assert csv_history_filepath == os.path.join(SAVE_FOLDER, id_string + '_history.csv')
    assert model_best_filepath == os.path.join(SAVE_FOLDER, id_string + '_model.pth')

    #Files should be created
    assert os.path.isfile(csv_history_filepath) == True
    with pytest.raises(OutputFileExistsException):
        csv_history_filepath, model_best_filepath = set_up_output_filepaths(testing,id_string,train_test_method)

    #Clean up 
    os.remove(os.path.join(SAVE_FOLDER, id_string + '_history.csv'))

def test_set_up_output_invalid_train_test_method():

    testing=False
    id_string = "1234"
    train_test_method = 10

    with pytest.raises(InValidTestTrainMethod):
        csv_history_filepath, model_best_filepath = set_up_output_filepaths(testing,id_string,train_test_method)


stanford_model_path = 'saves/320_model.pth'
def test_load_weight():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = construct_model(7, 196,0,0,0,0)
    model2 = construct_model(7, 196,0,0,0,0)

    assert model1 != model2

    model1 = load_weight(model1,stanford_model_path,device)
    model2 = load_weight(model2,stanford_model_path,device)

    #Checck models have been changed
    for model1_key, model2_key in zip(model1.state_dict(),model2.state_dict()):
        assert (model1.state_dict()[model1_key] == model2.state_dict()[model2_key]).all()

def test_load_weight_stanford():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = construct_model(7, 196,0,0,0,0)
    model2 = construct_model(7, 196,0,0,0,0)

    assert model1 != model2

    model1 = load_weight_stanford(model1,stanford_model_path,device)
    model2 = load_weight_stanford(model2,stanford_model_path,device)

    #Checck models have been changed
    for model1_key, model2_key in zip(model1.state_dict(),model2.state_dict()):
        assert (model1.state_dict()[model1_key] == model2.state_dict()[model2_key]).all()


def test_get_loss_function_cross_entropy():
    loss_function = get_loss_function(1)
    assert loss_function ==  F.cross_entropy

def test_get_loss_function_dual_cross_entropy():
    loss_function = get_loss_function(2)
    assert loss_function !=  F.cross_entropy

def test_get_loss_function_invalid():
    with pytest.raises(InvalidLossFunction):
        loss_function = get_loss_function(3)
