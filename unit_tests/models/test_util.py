import torch
from torch.autograd import Variable
import os

from datasets import prepare_loader
import torch.optim as optim


def check_layers_update(model):
    optimizer = torch.optim.Adam(model.parameters())

    #Same parameters before training
    list_parames = []
    for param in model.parameters(): # loop the weights in the model before updating and store them
        list_parames.append(param.clone())

    #Train model 
    inputs = Variable(torch.randn(1,3,244,244), requires_grad=True)
    targets = Variable(torch.zeros(1).long())
    optimizer.zero_grad()
    output = model(inputs)
    loss = torch.nn.functional.cross_entropy(output, targets)
    loss.backward()
    optimizer.step()

    index =0 
    for param in model.parameters(): # loop the weights in the model before updating and store them
        assert False == torch.equal(param,list_parames[index])
        index+=1

    return output

def check_layers_multitask(model):
    optimizer = torch.optim.Adam(model.parameters())

    #Same parameters before training
    list_parames = []
    for param in model.parameters(): # loop the weights in the model before updating and store them
        list_parames.append(param.clone())

    #Train model 
    inputs = Variable(torch.randn(1,3,244,244), requires_grad=True)
    targets = Variable(torch.zeros(1).long())
    
    optimizer.zero_grad()
    output = model(inputs)

    loss = torch.nn.functional.cross_entropy(output[0], targets)
    loss += torch.nn.functional.cross_entropy(output[1], targets)
    loss += torch.nn.functional.cross_entropy(output[2], targets)
    loss += torch.nn.functional.cross_entropy(output[3], targets)

    loss.backward()
    optimizer.step()

    index =0 
    for param in model.parameters():
        assert False == torch.equal(param,list_parames[index])
        index+=1

    return output


def check_classes_in_output(output,num_classes):
    for x,y in zip(output,num_classes):
        print(x.size()[1],y)
        assert x.size()[1] == y
