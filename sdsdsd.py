metrics = {
        'loss_meter':0,
        'acc_meter':1,
        'make_loss_meter':2,
        'make_acc_meter':3,
        'model_loss_meter':4,
        'model_acc_meter':5,
        'submodel_loss_meter':6,
        'submodel_acc_meter':7,
        'generation_loss_meter':8,
        'generation_acc_meter':9
    }

elapsed = 10
types='val'
saves = {
    types+'_loss': metrics['loss_meter'],
    types+'_acc': metrics['acc_meter'],

    types+'_make_loss': metrics['make_loss_meter'],
    types+'_make_acc':metrics['make_acc_meter'],

    types+'_model_loss': metrics['model_loss_meter'],
    types+'_model_acc':metrics['model_acc_meter'],

    types+'_submodel_loss':metrics['submodel_loss_meter'],
    types+'submodel_acc':metrics['submodel_acc_meter'],

    types+'generation_loss':metrics['generation_loss_meter'],
    types+'generation_acc':metrics['generation_acc_meter'],

    types+'_time': elapsed
}

print(saves)