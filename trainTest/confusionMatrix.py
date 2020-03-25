
import torch

def update_confusion_matrix(matrix,pred,targets):
    preds = torch.argmax(pred, 1)
    for p, t in zip(preds, targets):
        matrix[p, t] += 1