import torchvision
from models.NetworkVGG import NetworkV1,NetworkV1_1,NetworkV1_2,NetworkV1_3,NetworkV1_4,NetworkV1_5
from models.NetworkVGG import NetworkV2_ML_Boxcars1,NetworkV2_ML_Stan,NetworkV2_ML_Boxcars2,NetworkV2_ML_Boxcars3,NetworkV2_ML_Boxcars4,Network_Boxcars_Duplicate,
from models.NetworkVGG import Network_Boxcars_Duplicate_ML_One_FC,Network_Boxcars_Duplicate_ML_All_own_FC

# Set up config for other models in the future
def construct_model(config, num_classes,num_makes,num_models,num_submodels,num_generations):
    base = torchvision.models.vgg16(pretrained=True, progress=True)   
    if config['model_version'] == 1:
        model = NetworkV1(base, num_classes)
    elif config['model_version'] == 2:
        model = NetworkV2_ML_Boxcars1(base, num_classes,num_makes,num_models,num_submodels)
    elif config['model_version'] == 3:
        model = NetworkV1_1(base, num_classes)
    elif config['model_version'] == 4:
        model = NetworkV1_2(base, num_classes)
    elif config['model_version'] == 5:
        model = NetworkV1_3(base, num_classes)
    elif config['model_version'] == 6:
        model = NetworkV1_4(base, num_classes)
    elif config['model_version'] == 7:
        model = NetworkV1_5(base, num_classes)
    elif config['model_version'] == 8:
        model = NetworkV2_ML_Stan(base,num_classes,num_makes,num_models)
    elif config['model_version'] == 9:
        model = NetworkV2_ML_Boxcars2(base,num_classes,num_makes,num_models,num_submodels)
    elif config['model_version'] == 10:
        model = NetworkV2_ML_Boxcars3(base,num_classes,num_makes,num_models,num_submodels)
    elif config['model_version'] == 11:
        model = NetworkV2_ML_Boxcars4(base,num_classes,num_makes,num_models,num_submodels,num_generations)
    elif config['model_version'] == 12:
        model = Network_Boxcars_Duplicate(base, num_classes)
    elif config['model_version'] == 13:
        model = Network_Boxcars_Duplicate_ML_One_FC(base,num_classes,num_makes,num_models,num_submodels)
    elif config['model_version'] == 14:
        model = Network_Boxcars_Duplicate_ML_All_own_FC(base,num_classes,num_makes,num_models,num_submodels)
    print(model)
    exit()
    

    return model
