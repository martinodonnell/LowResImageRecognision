

from trainTest.testMethods import test_v1, test_v2, test_v3, test_v4,test_v5
from trainTest.trainMethods import train_v1, train_v2, train_v3, train_v4,train_v5
from trainTest.TrainTest6 import train_v6,test_v6
from trainTest.TrainTest7 import train_v7,test_v7

def get_train_test_methods(config):

    #Normal model
    if config['train_test_version'] == 1:
        print("Train/Test Version 1 for normal models")
        return train_v1, test_v1

    #---- Multitask learning --- 
    #Not used. Use model 6 and change multiplications on loss
    elif config['train_test_version'] == 2:
        print("Train/Test Version 2 for BOXCARS (Auxillary Multitask learning - 2 features) ")
        return train_v2, test_v2
    elif config['train_test_version'] == 3: 
        print("Train/Test Version 3 for STANFORD (Multitask learning)")
        return train_v3, test_v3
    # Not used. Use model 6 and change multiplications on loss
    elif config['train_test_version'] == 4:
        print("Train/Test Version 4 for BOXCARS (Multitask learning - 3 features)")
        return train_v4, test_v4
    elif config['train_test_version'] == 5:
        print("Train/Test Version 5 for BOXCARS (Multitask learning - 4 features)")
        return train_v5, test_v5
    elif config['train_test_version'] == 6:
        print("Train/Test Version 6 for BOXCARS (Auxillary learning - 4 features")
        return train_v6, test_v6
    elif config['train_test_version'] == 7:
        print("Train/Test Version 7 for normal models dual cross entropy")
        return train_v7, test_v7
    else:
        print(config['train_test_version'], "is not a valid trainTest method(get_train_test_methods)")
        exit(1) 