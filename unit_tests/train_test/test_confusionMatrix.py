
import torch
import sys
sys.path.append("..")
from train_test.confusion_matrix import update_confusion_matrix

print(sys.path)

def test_update_confusion_matrix():
    num_classes = 5

    pred = torch.tensor([[0,1,2,3,4]])
    target = torch.tensor([4])
    matrix = torch.zeros(num_classes, num_classes)

    update_confusion_matrix(matrix,pred,target)

    assert matrix[4,4] == 1