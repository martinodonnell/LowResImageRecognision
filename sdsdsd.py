types = 'train'
saves = {
    types+'_loss': ['loss_meter'],
    types+'_acc': ['acc_meter'],

    types+'_make_loss': ['make_loss_meter'],
    types+'_make_acc':['make_acc_meter'],

    types+'_model_loss': ['model_loss_meter'],
    types+'_model_acc':['model_acc_meter'],

    types+'_submodel_loss':['submodel_loss_meter'],
    types+'submodel_acc':['submodel_acc_meter'],

    types+'generation_loss':['generation_loss_meter'],
    types+'generation_acc':['generation_acc_meter']
    
}

print(saves['train_loss'])